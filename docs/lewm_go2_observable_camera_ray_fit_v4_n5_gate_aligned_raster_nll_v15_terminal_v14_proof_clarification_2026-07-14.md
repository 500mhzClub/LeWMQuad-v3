# Observable Camera-Ray Fit V4 N5 Gate-Aligned Raster-NLL V15 terminal-V14 proof clarification

Date: 2026-07-14

Status: **source-free proof-closure clarification only; no experiment authority**

## Trigger

The governing V15 runtime-visibility successor amendment is:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_runtime_visibility_successor_amendment_2026-07-14.md`

Its file SHA-256 is
`b1809b74cd400f8c56b5a912017c9466bb69aa0a7f4e390ccd3be59492a0f393`.
It correctly preserves the terminal V14 attempt and forbids its deletion,
mutation, rename, repair, retry, or reinterpretation. It also requires the
retained V14 suite to pass `235/235` during V15 implementation and review.

Those two requirements became mutually incompatible only after the authorized
V14 attempt was consumed. The immutable test

`lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py::test_v14_sources_forbid_v11_checkpoint_input_and_exact_root_is_absent`

contains two assertions: the V14 sources must not consume a V11 checkpoint,
and the not-yet-executed V14 output root must be absent. The first assertion
remains valid. The second was a pre-attempt freshness assertion and now fails
for the correct reason: the terminal V14 reservation and failure evidence must
remain present forever.

The terminal V14 evidence is exactly:

| Role | File SHA-256 | Canonical content SHA-256 |
|---|---|---|
| `attempts/seed_20260710/n5/reservation.json` | `56abce14d8ba7901103bbd23353095c30180ca5361f7595b178da6e440ecea8c` | `0aabc7ac8a468c6524ba66a244a11126c8e2c1d7587dbc1fafb3de71cc7d443b` |
| `attempts/seed_20260710/n5/failed.json` | `df6d91925fb167bc72e41eb9a6f07657f246c6a7a95d3bf20734c747e639c704` | `79560d4b5532ff41e428da913e28c6db235608da6cf4ea107fd33207870afad7` |

The only other file in the V14 root is the exact zero-byte reservation lock at
`attempts/seed_20260710/.n5.reservation-v14.lock`, whose SHA-256 is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
No checkpoint, result, completion, metric-verification, or gate artifact exists.

## Exact proof replacement

V15 implementation and independent review must not modify the V14 test or any
V14 source/output byte. They must replace only the now-obsolete retained-suite
expectation as follows:

1. run the complete immutable V14 suite while deselecting exactly the one fully
   qualified test node named above;
2. require all remaining `234` V14 tests to pass and require exactly one test,
   that exact node and no other node, to be deselected;
3. independently reproduce the still-live source assertion from the deselected
   test by proving the forbidden V11 attempt path is absent from the exact V14
   trainer, verifier, and executor source bytes;
4. validate the V14 root through no-follow regular-file reads and require its
   relative-file inventory to equal exactly the reservation, terminal failure,
   and zero-byte lock paths above;
5. rehash the reservation, failure, and lock to the exact file identities above
   and validate both JSON objects as canonical one-value-plus-LF documents with
   their exact self-content hashes;
6. require reservation `attempt_index=1`, `maximum_attempts=1`, seed `20260710`,
   fit size `5`, and the canonical V14 source-review binding;
7. require terminal `status=failed`, `failure_stage=training`,
   `failure.class=runtime`, `failure.code=execution_failure`,
   `partial_artifacts_removed=true`, intact owned-directory journal, and
   `retry_authorized=false`;
8. require every checkpoint, new-model, metric, later-rung, Shared-JEPA,
   selection, calibration, G2, navigation, held-out, runtime, hardware,
   production, promotion, deployment, and retry authority in the terminal
   record to be false; and
9. require absence of V14 `checkpoint.pt`, `result.json`, `completed.json`,
   metric-verification, and gate paths.

The author and reviewer must report this retained proof as:

`V14: 234 passed, 1 exact pre-attempt-absence node deselected, 1 terminal-state replacement proof passed`.

It must not be reported as `235/235`, as an ordinary skip, or as an unexplained
failure. Any other V14 failure, deselection, inventory member, hash, semantic
value, authority value, or missing terminal artifact is a V15 source-review
`BLOCK`.

The V15 proof closure must bind and rehash this clarification in addition to
the governing V15 amendment. The implementation handoff and independent review
must name both documents and their exact hashes. The V15 source policy must
include this clarification in its frozen source bindings so changed or absent
bytes reject before any runtime diagnostic or exact action.

## Preserved requirements and non-authority

This clarification changes no V14 or V15 model, tensor operation, loss,
coefficient, data selection, RGB mapping, schedule, seed, optimizer, update,
threshold, metric, control, verifier, lifecycle rule, runtime-visibility rule,
receipt rule, namespace, attempt count, or output path. It creates no new V14
attempt and never converts the V14 infrastructure failure into a numeric
result.

All other V15 amendment proofs remain required, including V15, V13, V12, and
V11 suites, normalized-AST parity, visibility dispositions, no-open spies,
pre-reservation quiescence, and lifecycle fault injection.

This file authorizes only V15 source/proof/handoff adjustment and subsequent
different-agent source review. It authorizes no GPU diagnostic, data or RGB
open, `/tmp` visibility receipt, V15 reservation or exact execution, V14 retry
or mutation, later rung, training, G2, navigation, development, held-out,
runtime, hardware, production, promotion, or deployment action.
