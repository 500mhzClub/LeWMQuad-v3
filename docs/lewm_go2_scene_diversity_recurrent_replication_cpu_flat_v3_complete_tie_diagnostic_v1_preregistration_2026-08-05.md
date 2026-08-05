# CPU-flat V3 post-hoc complete-tie diagnostic V1

Date frozen: 2026-08-05

Status before execution: `PREREGISTERED_NONCONFIRMATORY_DIAGNOSTIC_NOT_RUN`

This document preregisters one evaluation-only diagnostic successor to the
consumed CPU-flat V3 scientific attempt. It does not retry, resume, repair,
complete, or salvage that attempt. The predecessor remains terminal with no
scientific decision.

## Question and scope

The sole question is whether the already-trained frozen recurrent checkpoint
can be evaluated on all 128 existing evaluation states after completing the
mathematical domain of the frozen dense-rank scorer for a complete tie.

The predecessor failed before model evaluation because four evaluation states
had exact dense ranks `[0, 0, 0, 0, 0, 0, 0, 0, 0]`. Its independent terminal
review found no collection or checkpoint-integrity failure, localized those
four states exactly, and authorized no successor. This diagnostic requires a
fresh root and a separate independent source review.

## Exact predecessor evidence

The diagnostic plan must bind and rehash exactly these five inputs before its
fresh root is created:

| Input | SHA-256 | Bytes |
|---|---:|---:|
| `docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_scientific_exact_plan_2026-08-05.json` | `0ad79cc46cead469d6532cd0be04c5d7623fffe18ddafc737c32855d6c9a8f29` | 359,692 |
| `.generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3/attempt_v1/terminal.json` | `a4da81177d77372923b72775f69cfe58b596a651017ef6ebc5988df05d390327` | 1,273 |
| `.generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3/attempt_v1/collection/physics_result.json` | `711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0` | 369,067 |
| `.generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3/attempt_v1/checkpoint.pt` | `6c16f97ae5748e1d230244b4588f3efc11330a2673bd15e2ff83aa2f2392844e` | 167,423 |
| `docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_scientific_terminal_review_2026-08-05.json` | `7218c78387871e82280f96fe746acb047f46d1a2836b7638b12ce9c1514a81dd` | 17,379 |

The predecessor `result.json` must remain absent. The diagnostic may not use
the qualification payload, another checkpoint, another collection, or an
unbound replacement for any input above.

## Sole scoring-domain completion

For every state, let `r` be its nine frozen integer dense ranks and let
`m = max(r)`. A complete tie is defined generally and only as `m == 0`, which
given the unchanged nonnegative-rank invariant means all nine ranks are zero.

The diagnostic makes exactly these scoring definitions:

- the normalized-rank denominator is `max(1, m)`;
- for a complete tie, every one of the nine actions is oracle-equivalent;
- every selected action in a complete tie has normalized rank regret `0`;
- uniform-random expected regret in a complete tie is `0`; and
- uniform-random oracle-equivalent selection rate in a complete tie is `1`.

For every state with `m > 0`, selection, regret, random expectation, oracle
equivalence, physical summaries, aggregation, and all other scorer behavior
remain byte-for-scientific-behavior identical to the frozen evaluator. The
one-centimetre rank tolerance remains unchanged. There is no epsilon,
near-tie rule, floating tolerance change, rank recomputation, or scene-specific
exception.

## Frozen evaluation science

The following remain unchanged:

- all 128 evaluation states, including all four complete-tie states;
- the 32-scene/eight-family evaluation balance and original order;
- the durable checkpoint identity and its six completed learned members;
- learned arms `no_vision_recurrent_direct` and
  `visual_recurrent_direct`;
- model seeds `2026080411`, `2026080412`, and `2026080413`;
- 800-update checkpoint contents, model equations, projections, and input
  statistics;
- the live `task_action_only`, privileged physical oracle, and uniform-random
  reports;
- CPU DINOv2 context extraction from the existing evaluation context only;
- the 10,000-resample paired family/scene bootstrap;
- the original five gates and thresholds: visual regret at most `0.13`, visual
  minus task/action at most `-0.02`, visual minus no-vision at most `-0.01`,
  both paired upper 95% bounds below zero, and visual better than random;
- two evaluations from the same durably reopened checkpoint with exact result
  equality; and
- zero successor-observation access.

No learned member is trained, updated, calibrated, selected, or replaced.
Existing train-role receipt metadata may be reopened only where the frozen
evaluator requires it to reconstruct the unchanged live analytic
`task_action_only` control and verify checkpoint-linked identities. No train
context feature extraction or learned-model fitting is permitted.

## Prohibited changes

This diagnostic permits no Genesis execution, rerender, recollection, scene
filtering, state filtering, refill, changed action set or horizon, changed
rank tolerance, changed threshold, changed bootstrap, additional seed,
checkpoint modification, learned-model retraining, retry, resume, overwrite,
repair, or second invocation.

No sealed, held-out, production, or other protected material may be opened,
searched, copied, or used.

## Result interpretation

The output is post-hoc, development-only, and non-confirmatory. Mechanical
pass/fail values for the unchanged gates may be reported as diagnostic
measurements, but they do not become the missing CPU-flat V3 scientific
decision and cannot support a navigation, planning, representation,
world-model, generalization, promotion, or deployment claim.

The fresh root is:

`.generated/dev/go2_scene_diversity_recurrent_replication_cpu_flat_v3_complete_tie_diagnostic_v1/attempt_v1`

Before that root is created, an independent review must bind this
preregistration, exact diagnostic plan, plan builder, diagnostic runner,
focused tests, the five predecessor inputs, and the complete-tie adapter. Its
schema/status are:

- `lewm_go2_scene_diversity_recurrent_replication_cpu_flat_v3_complete_tie_diagnostic_v1_source_review_v1`;
- `PASS_INDEPENDENT_COMPLETE_TIE_DIAGNOSTIC_SOURCE_REVIEW`.

That review may clear exactly one diagnostic invocation under the user's
standing instruction. It creates no retry, resume, scientific-confirmation,
V3-salvage, promotion, or deployment authority.
