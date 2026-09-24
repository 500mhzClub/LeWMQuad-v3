# Go2 RGB multiresolution perception V2 operational-recovery decision

Date: 2026-07-24

Author: `/root/terminal_failure_audit/retry_authority_review`

Status: **SELECTED SCIENCE-IDENTICAL OPERATIONAL RECOVERY; SOURCE-ONLY
PREREGISTRATION AUTHORITY; NO GENERATED-INPUT, CHECKPOINT, GPU, TRAINING,
QUALIFICATION, G2, NAVIGATION, HELD-OUT, PRODUCTION, PROMOTION, OR DEPLOYMENT
AUTHORITY**

## Exact user authorization

> Authorize exactly one science-identical V2 integrity replacement. Change only the schedule-schema adapter and complete failure receipts; preserve the model, data, seed, schedule, losses, thresholds, initialization, and 16,000-presentation cap.

This is the narrow scientific exception that the V1 terminal audit found was
required. It authorizes source work and preregistration for one separately
versioned V2 operational recovery. It does not revive, retry, resume, recover,
unseal, delete, rename, or mutate V1, and it is not itself V2 execution
authorization.

## Audited V1 predicate

The governing audit is
`docs/lewm_go2_rgb_multiresolution_perception_v1_terminal_lifecycle_failure_audit_2026-07-24.json`
at commit
`e3e0cc50877c9dc5cbd7d269e4b169f19857e897`:

- file SHA-256:
  `6adaaaea3ec1d63438f63e5282b832c27c34348075f57317070acd04b615b541`;
- canonical content SHA-256:
  `ccfc14731e569aed773d4380865395a60e00d8354ba9903757b1f23675a7b3d3`;
- byte count: `7,363`;
- status:
  `INDEPENDENT_AUDIT_CONFIRMS_SEALED_TERMINAL_INTEGRITY_FAILURE_WITH_ACCESS_RECEIPT_GAP`.

The audit binds V1 attempt identity
`6caf034472ef220564bec9116dbe2b64b25ad1d36072d8a11056224da582ba3f`
and its execution authorization:

- path:
  `docs/lewm_go2_rgb_multiresolution_perception_v1_execution_authorization_2026-07-24.json`;
- file SHA-256:
  `522cba9cefed795cfd03b9db3949881a65fe24620821bc463a96a7920326c542`;
- canonical content SHA-256:
  `cb06d8642484e95030fc9ce26b57f2efe60b7977ebb99ae1373321b97d9551ed`;
- byte count: `7,834`.

V1 reserved and consumed
`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_multiresolution_perception_probe_v1`.
That root is sealed and contains exactly `reservation.json` and `failed.json`.
It has no `result.json`, `completed.json`, `access.json`, probe checkpoint, or
scientific result. The independent audit establishes:

- zero complete optimizer updates;
- zero pair-index presentations;
- zero backward calls and Camera or JEPA objectives;
- zero checkpoint-selection evaluations;
- zero RGB payload opens; and
- training was never entered.

V1 remains a consumed terminal lifecycle attempt. This decision does not
reinterpret it as unused.

## Decision

Authorize source implementation and independent review of exactly one fresh
V2 attempt under a new absent output root:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_multiresolution_perception_probe_v2`

V2 must initialize fresh from the same bound N320 initialization input. It may
not load any V1 runtime output, resume state, checkpoint, metric, or partial
state. Reservation of the fresh V2 root consumes its sole attempt. V2 has no
retry, resume, recovery, second seed, extension, observer rerun, or replacement
attempt.

## Exact scientific identity

The complete unchanged scientific contract is embedded in
`docs/lewm_go2_rgb_multiresolution_perception_v2_preregistration_2026-07-24.json`.
Its canonical SHA-256 is:

`e181381c00585fa5df41a71fff918b5599acc955d59283ce397ba6dd530dc23f`

Equality is defined mechanically:

1. parse the canonical committed V1 execution authorization;
2. extract its top-level `experiment` object;
3. encode that object with JSON keys sorted, separators `,` and `:`, ASCII
   escaping, and non-finite numbers forbidden;
4. SHA-256 the encoded bytes;
5. require the embedded V2 `science_contract` object to be deeply equal to the
   extracted V1 object and to produce the same digest above.

The embedded contract deliberately retains the V1 model-family and runtime
version identifiers. V2 versions only the operational execution envelope.
Model architecture and parameters, N320 migration, data roles and order,
presentation indices, seeds, optimizer, learning-rate function, losses,
coefficients, thresholds, evaluation checkpoints, operation counts, and the
16,000-presentation cap are unchanged.

## The only two allowed code deltas

### 1. Bound schedule-schema adapter

Replace only the failing schedule-schema adaptation boundary. The V2 adapter
must validate the already-bound matched-training schedule under its owning
canonical schema, validate its exact file and content bindings, validate the
unchanged ordered train-role identity and full schedule invariants, and expose
the same first 16,000 integer presentation indices without mutation,
regeneration, reordering, filtering, or reseeding. It must recheck the frozen
prefix hashes at 1,600, 6,400, and 16,000 presentations.

Schedule validation moves to the first post-reservation runtime-input stage,
before opening the N320 tensor checkpoint or raw supervision indexes. This
ordering change is operational custody only; it cannot change the schedule or
the scientific payload.

### 2. Complete failure receipts

Add a durable partial-access ledger that is published before or atomically with
every post-reservation generated-input open and updated with the observed
binding and outcome. Every terminal failure must:

- directly bind `reservation.json` by relative path, file SHA-256, canonical
  content SHA-256, and byte count;
- bind the latest partial-access ledger and enumerate the exact attempted and
  completed post-reservation opens;
- publish exact zero-or-observed operation counts, failure stage and error;
- retain the attempt identity and all downstream denials; and
- seal every terminal file read-only and every terminal directory read-only.

Injected-failure tests must cover the schedule, gate, N320 checkpoint, raw
authority, raw indexes, model preparation, training, evaluation, result
publication, and completion-publication boundaries.

No third code delta is authorized. In particular, source cleanup, refactoring,
model changes, data repair, seed or order changes, loss changes, threshold
changes, schedule extension, performance tuning, and dependency upgrades are
outside scope.

## Required custody order

Before any V2 generated-input, checkpoint, GPU, or training access:

1. commit this decision and the canonical V2 preregistration;
2. obtain an independent preregistration/recovery-exception review;
3. implement only the two allowed operational deltas and their tests;
4. produce a new recursive V2 source manifest;
5. perform an allowlist-only clean export and certification;
6. obtain an independent source and lifecycle review;
7. create a distinct execution authorization binding all preceding artifacts,
   the exact V1 science-contract digest, runtime inputs, hardware contract, and
   absent V2 output root; and
8. run one fail-closed preflight followed immediately by the one V2 attempt.

The implementation author, independent source reviewer, and execution
authorizer must remain distinct. V1 stays sealed throughout.

## Downstream authority

Neither this decision nor the V2 preregistration qualifies a checkpoint or
authorizes perception qualification, JEPA training, G2, navigation, held-out
access, production, promotion, or deployment. A V2 terminal PASS may authorize
only a separately preregistered bounded perception-qualification attempt, as
specified by the unchanged scientific contract. V2 terminal failure, including
an integrity failure, authorizes nothing further and has no retry.
