# RGB causal temporal perception V1 terminal audit

Date: 2026-07-24

## Scope and identities

This is a source-only, metadata-only terminal audit. It grants no execution,
retry, checkpoint, qualification, JEPA, G2, navigation, held-out, production,
promotion, or deployment authority.

- Preregistration commit:
  `3e30b8ae9dbdfeafd0f62bfc4243cece7a885d95`.
- Frozen implementation commit:
  `75240453b69cbbe34e6dbbdd5e65765aba7d26e6`.
- Review and authorization commit:
  `fd30f03d2bbf19d6478ea52a898517dc1a227299`.
- Independent source review:
  `docs/lewm_go2_rgb_causal_temporal_perception_v1_source_review_2026-07-24.json`,
  file SHA-256
  `cfbc1f09d62c734ca9658454ff4fc82085eca51574b534556927b095f9a6597d`,
  content SHA-256
  `1d6fbce1c77b6315268b01fa241eabec6cf7aa895f6fcd2e13d86d789d1b46cd`,
  16,277 bytes, reviewer `/root/frozen_source_review`,
  `PASS_SOURCE_ONLY`.
- Execution authorization:
  `docs/lewm_go2_rgb_causal_temporal_perception_v1_execution_authorization_2026-07-24.json`,
  file SHA-256
  `940dccd46dced944619b35eea2a72711f6820537c19d04776cef0fdcd6345ed0`,
  content SHA-256
  `a95b0584f698bde20a7179ab8cd2bb8c01b25666aa85bbd1c2f92ef6c46bc7b0`,
  12,279 bytes, authorizer `/root/pair_timing_hardening`.
- Attempt root:
  `.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_causal_temporal_perception_probe_v1`.
  Attempt identity:
  `6d0422241f1d3f7e057230ad679bbf70a78d4b2f719cfbf1030a118280afd1f6`.
- Immutable metric index:
  `checkpoint_metrics.json`, file SHA-256
  `8a17da6165b26c42743ac1b438d9082deac2f9c71c75af02aabda5024f63bd7f`,
  content SHA-256
  `9883a6350f5921a611c09284f90d1c25ff20dc600bd7c9d0df2574fd16091223`,
  72,156 bytes.
- Partial-access ledger:
  `partial_access.jsonl`, file SHA-256
  `96dd9307918e1087f444a6a2dc8c86e4788fd7171ff6ec70bf6ba0552a2f2ad3`,
  26,975,088 bytes.
- Terminal result:
  `result.json`, file SHA-256
  `d022e2a2e0210d7ea5af1b3d3ea13c454d4e283a013c29c9f7b8d997c3bc86fc`,
  content SHA-256
  `1b5c4f9b5c7f66c5a68077c88d9f8ddb69e1cbc2e387b70fbc38f34ba5f9aab7`.

No checkpoint tensor payload was opened for this audit.

## Observed checkpoints

Updates 100 and 400 were informational. Update 1,000 was the sole terminal
decision.

| Update | Complete scopes | Passed margins | Total shortfall | Rough pixel balanced accuracy | Rough ground balanced accuracy | Rough depth p95 m | Control |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 0 | 27 / 189 | 141.38868579605398 | 0.5210038024592916 | 0.6082898683636179 | 2.785243320465087 | `CONTINUE_INFORMATIONAL` |
| 400 | 0 | 84 / 189 | 60.30224227461466 | 0.6770213896624805 | 0.6124017445238973 | 1.3475676536560057 | `CONTINUE_INFORMATIONAL` |
| 1,000 | 0 | 111 / 189 | 33.13261634065992 | 0.7403405148373643 | 0.6217081280253147 | 1.0263007879257195 | `FAIL_TERMINAL_NO_RETRY` |

The run completed 1,000 optimizer updates and the full capped 16,000 pair
presentations.

## Terminal conjunction at update 1,000

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Complete physical scopes | at least 1 | 0 | FAIL |
| Passed margins | at least 98 / 189 | 111 / 189 | PASS |
| Total shortfall | strictly below 41.01776266878769 | 33.13261634065992 | PASS |
| Rough pixel balanced accuracy | strictly above 0.8198594673963917 | 0.7403405148373643 | FAIL |
| Rough ground balanced accuracy | strictly above 0.647134926562893 | 0.6217081280253147 | FAIL |
| Rough depth p95 m | strictly below 0.9777327477931971 | 1.0263007879257195 | FAIL |

Four of the six mandatory conjuncts failed. The learned causal temporal
residual did not qualify perception.

## Comparison with multiresolution V3

The prior multiresolution V3 terminal result at update 1,000 was:

- 0 complete scopes;
- 111 / 189 passed margins;
- total shortfall `33.247456241393685`;
- rough pixel balanced accuracy `0.7415505748284441`;
- rough ground balanced accuracy `0.6220155782326704`;
- rough depth p95 m `1.0238167285919189`.

Temporal V1 retained exactly the same scope and margin counts. Shortfall
improved by about `0.115`, while all three rough-motion metrics became
slightly worse. It therefore produced essentially no useful benefit over V3
and failed the same four terminal gates.

## Strict receipt status

The official frozen
`contract.parse_partial_access_ledger(partial_access.jsonl)` rejects the
ledger with:

```text
PermissionError: attempted input escaped frozen runtime roots
```

The first rejection is record sequence 17, open ID 9:

```text
.generated/go2_render_selected_v04/scenes/scene_bc5a05ec9fce8d9c/rgb/frame_026900_env_20.png
```

It is a `development_rgb` open for role `train`, stage
`training_batch_materialization`, purpose `runtime_load`. The frozen
post-hoc validator allows five fixed runtime leaves and paths below the raw
supervision root, but omits the separately authorized development-render RGB
root. The execution authorization explicitly allowed development RGB decode
for the train and checkpoint-selection roles.

The strict formal status is therefore **contract-invalid and inadmissible**:
this attempt cannot serve as contract-valid qualification evidence.

## Forensic ledger replay

A read-only forensic replay classified every path without opening any
referenced payload:

- 37,714 total ledger records;
- 18,856 open attempts and 18,856 paired outcomes;
- all 18,856 outcomes recorded `ACCEPTED`;
- 17,402 attempts outside the validator's fixed/raw-root allowlist;
- exactly 8,701 unique outside paths, each appearing twice: one runtime load
  and one terminal rehash;
- all 17,402 matched exactly
  `.generated/go2_render_selected_v04/scenes/scene_[0-9a-f]{16}/rgb/frame_[0-9]{6}_env_[0-9]{2}.png`;
- zero pattern mismatches;
- 7,777 train runtime loads across 72 scenes;
- 924 checkpoint-selection runtime loads across 8 scenes;
- 8,701 authority-scoped terminal rehashes.

With only a narrow in-memory adapter admitting that exact path pattern, the
unchanged full ledger passes canonical-record, self-hash, chain, pairing,
outcome, operation-count, and terminal-finalization validation. This
identifies a post-hoc validator root-allowlist defect, not evidence of
unauthorized experiment access. It does not make the receipt formally valid,
and the frozen source and completed output must remain untouched.

## Disposition and authority boundary

The robust disposition is also terminal: the checkpoint is not qualified,
the mechanism is terminated, and no retry is warranted because the observed
metrics fail decisively even if the receipt defect is set aside.

Recorded counters remain zero for JEPA objectives, JEPA backward passes, EMA
updates after initial hard sync, probability calibration, G2, navigation,
held-out access, prior runtime-output access, and rejected/protected
adaptation-checkpoint access. No held-out or sealed material was accessed.

A materially different next hypothesis is learned post-encoder alignment
conditioned on admissible causal ego-motion before temporal fusion. That is a
proposal only. It requires a new preregistration, independent source review,
and explicit execution authorization; this audit grants none.
