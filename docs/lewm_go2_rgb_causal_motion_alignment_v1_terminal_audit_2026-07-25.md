# RGB causal motion alignment V1 terminal audit

Date: 2026-07-25

## Scope and identities

This is a source- and metadata-only terminal audit. It grants no execution,
retry, checkpoint, qualification, JEPA, G2, navigation, held-out, production,
promotion, or deployment authority.

- Preregistration commit:
  `a3cea116e5cdf6cfec3801624c51306742e0f0f5`.
- Frozen implementation commit:
  `d5be6db159b38697ac87b96ccec4c5871042c040`.
- Independent review commit:
  `32da42ae0b9b664059fe882bc6a72c7d80dcae69`.
- One-attempt authorization commit:
  `94ec9c122a99453632a35b14f08c737831d99172`.
- Independent source review:
  `docs/lewm_go2_rgb_causal_motion_alignment_v1_source_review_2026-07-24.json`,
  file SHA-256
  `750894fc5bf225da38d0cf62e72a88b3a2a94c58f636b8a49d5b8acbc6491396`,
  content SHA-256
  `7b723feee1d56438a8672af9660d76820b68048e26dfdde2bf002bbb70a67cd3`,
  20,120 bytes, reviewer `/root/runner_adapter_review`,
  `PASS_SOURCE_ONLY`.
- Execution authorization:
  `docs/lewm_go2_rgb_causal_motion_alignment_v1_execution_authorization_2026-07-24.json`,
  file SHA-256
  `54cee192cee4a0e61802b4d1b0930f08aa4b17b602155a8b5e73b7c907344b83`,
  content SHA-256
  `484ae4debe9e7ce3f551ba68e2ca071d711fd8a3c2ea529428b294b43af74c32`,
  13,827 bytes, authorizer `/root/authorization_schema_audit`.
- Attempt root:
  `.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_causal_motion_alignment_probe_v1`.
  Attempt identity:
  `736839515ee85d42ddcb2840baaef1310f5e971bbaa4d86ae386494c104b5955`.
- Immutable metric index:
  `checkpoint_metrics.json`, file SHA-256
  `de6106a601c198125ad835ecae746e7ef0e159dff16b481188f4b593cf83484f`,
  content SHA-256
  `d3e6b3c8b2d1b3f7b4287e885ddcd56505f09f55610147e17d95de82711cff86`,
  69,036 bytes.
- Partial-access ledger:
  `partial_access.jsonl`, file SHA-256
  `9262ed61939aa65c63864253ded8e4c4504b9ef67004d059a27ee9074d6f3bcf`,
  26,861,946 bytes.
- Terminal result:
  `result.json`, file SHA-256
  `e919bac25f6f0c7d0f640605b1a36b5cdd6189bb7d70358c5288a816982224c0`,
  content SHA-256
  `8a63568bd0af0dac12eb7032190ffbfd6cda6f12db078c8c80e178c3bff8b1fc`,
  13,243 bytes.
- Completion receipt:
  `completed.json`, file SHA-256
  `00f45e0abffd01836347c8a15f3e860b3c59229bd25be871a826f6ee52f033cd`,
  content SHA-256
  `91cdc93b6f394464149a9c7d2102b02116fa22cd2164e02e592293c39f3f2aba`,
  5,738 bytes.

No checkpoint tensor payload was opened for this audit.

## Observed checkpoints

Updates 100 and 400 were informational. Update 1,000 was the sole terminal
decision.

| Update | Complete scopes | Passed margins | Total shortfall | Rough pixel balanced accuracy | Rough ground balanced accuracy | Rough depth p95 m | Control |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 0 | 27 / 189 | 141.38720543849027 | 0.5209942194112362 | 0.6083129953074762 | 2.7854034423828122 | `CONTINUE_INFORMATIONAL` |
| 400 | 0 | 84 / 189 | 60.31282786143973 | 0.6770098563501225 | 0.6123407587438711 | 1.3472193002700803 | `CONTINUE_INFORMATIONAL` |
| 1,000 | 0 | 111 / 189 | 33.05143763708337 | 0.741796837511955 | 0.621981002078303 | 1.0227776646614073 | `FAIL_TERMINAL_NO_RETRY` |

The run completed exactly 1,000 optimizer updates and the full capped 16,000
pair presentations. It performed 4,000 camera objectives and backward calls,
with zero JEPA objectives, JEPA backward calls, or EMA updates after the
initial hard sync.

## Terminal conjunction at update 1,000

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Complete physical scopes | at least 1 | 0 | FAIL |
| Passed margins | at least 98 / 189 | 111 / 189 | PASS |
| Total shortfall | strictly below 41.01776266878769 | 33.05143763708337 | PASS |
| Rough pixel balanced accuracy | strictly above 0.8198594673963917 | 0.741796837511955 | FAIL |
| Rough ground balanced accuracy | strictly above 0.647134926562893 | 0.621981002078303 | FAIL |
| Rough depth p95 m | strictly below 0.9777327477931971 | 1.0227776646614073 | FAIL |

Four of the six mandatory conjuncts failed. The learned motion-alignment
mechanism did not qualify perception.

## Comparison with causal temporal V1

Temporal V1 ended with 0 complete scopes, 111 / 189 passed margins, total
shortfall `33.13261634065992`, rough pixel balanced accuracy
`0.7403405148373643`, rough ground balanced accuracy `0.6217081280253147`,
and rough depth p95 `1.0263007879257195` m.

Motion alignment retained exactly the same scope and margin counts. Its
changes were very small:

- total shortfall improved by about `0.08118`;
- rough pixel balanced accuracy improved by about `0.001456`;
- rough ground balanced accuracy improved by about `0.000273`;
- rough depth p95 improved by about `0.003523` m.

Although its metrics were still improving between fixed checkpoints, under
the fixed schedule and 16,000-presentation cap the new mechanism ended at
essentially the same rejected endpoint and failed the same four gates. This
is evidence that this preregistered post-encoder motion-alignment mechanism
did not remove the limiting bottleneck; it does not establish that all
post-encoder motion-alignment mechanisms would fail.

## Strict receipt status

The corrected successor parser accepts the complete ledger:

- 37,714 canonical, self-hashed, chained records;
- 18,856 attempted opens and 18,856 paired outcomes;
- all 18,856 outcomes accepted, with zero rejected or failed opens;
- final record `RUNTIME_INPUT_ACCESS_FINALIZED`;
- all consumed inputs rehashed;
- only `train` and `checkpoint_selection` development roles opened;
- zero prior-runtime-root, rejected-checkpoint, probability-calibration,
  G2, navigation, or held-out opens.

The exact terminal inventory contains 13 files. All files are mode `0444`,
both directories are mode `0555`, and no symlink is present. The terminal
integrity status is **PASS**.

## Disposition and authority boundary

The official result is
`FAIL_BOUNDED_FALSIFICATION_MECHANISM_TERMINATED`, with
`integrity_pass=true`, `qualifies_probe=false`, and
`retry_authorized=false`.

The checkpoint is not qualified. This mechanism is terminated, and its sole
attempt is consumed. It authorizes no perception qualification, JEPA
training, G2, navigation, held-out access, checkpoint reuse, retry, repair,
second seed, extension, promotion, production, or deployment.

The result argues against another post-encoder temporal-alignment tweak. A
materially different next hypothesis should target how spatial evidence is
represented before the unchanged evidence head, potentially through an
encoder-level mechanism. That is a proposal only and requires a new
preregistration, independent review, source freeze, and explicit execution
authorization.
