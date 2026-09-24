# V4 N5 full-panel V7 independent review

Date: 2026-07-13

Reviewer: `/root/camera_v5_independent/camera_v7_pre_freeze_review`

Verdict: **PASS**

The frozen V7 package satisfies the owned-directory transaction amendment and
closes the two terminal V6 directory-history findings without changing the
numerical experiment. The implementation author is
`/root/camera_v5_independent`, so this is a different-agent review.

No frozen author file was modified. Review work was CPU-only with HIP, CUDA,
ROCr, and ordinal visibility empty and native math threads capped at one. It
did not run exact execution or optimization, open a GPU, or open dataset, RGB,
model, checkpoint, protected-role, G2, held-out, runtime, hardware, navigation,
production, or promotion payloads. The canonical V7 experiment output remained
absent throughout review.

## Frozen closure

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v7_owned_directory_transaction_amendment_2026-07-13.md` | `17ca6b726d1eaa25662a1823b4c153d496f1e51502b764350ddd6a3a34f249da` |
| `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v7_implementation_handoff_2026-07-13.md` | `020a26678670ac0067a090a2f3c4ba3634185f4a450c48a24c657cd263c9b6be` |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v7.py` | `ed50a00c0449c41031f076c5627f6501b93ee2931deaf4cbcd06a0f9e89d16e0` |
| `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v7.py` | `5043d42aaabb5a4852e9339a7d3e98c9d530c7ff403e5a2f1ac7a21999fbc14e` |
| `lewm/tests/n5_full_panel_v7_synthetic_execution.py` | `9743786550ede91023b3d96cfa6650c04bd02a2c1a5d3fbb2364728b09980bf1` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v7.py` | `0bf0f77ff5c773891ddd6ab5ed933b74132f0c8194e0aa237d93175619b7a858` |

`preflight_static_authority()` independently rehashed the complete retained
V1-V6 authority closure, all four earlier BLOCK records, the V6 terminal BLOCK,
the admitted V5 source review, and the V5 reservation and terminal failure
receipts. It confirmed that no V5 numeric payload survived and no retry is
authorized. The V6 BLOCK canonical content remains
`98260f2b1af7845af6cf1312698b7a5c0d6a0579705f4ff522801eaa02d41fb1`.

## Transaction findings

### Event provenance and committed state

PASS. Every mutation below the exclusive V7 root is performed through a
retained descriptor and a closed operation-scoped journal transaction. Each
transaction drains prior events, captures all watched pre-state, rejects a
snapshot race, performs one declared operation, captures the post-state,
requires the exact inotify watch/name/mask/order/cookie sequence, proves the
declared inventory delta, and commits only the already captured post-state.

There is no generic refresh or mutable-descriptor acceptance path. Overflow,
unknown or reused watches, unexpected names or masks, cookie/order mismatch,
unmount, and unpaired self events permanently poison success. Adaptations of
both frozen V6 create/delete blockers reject while ordinary owned mutations
and unrelated shared-ancestor churn pass.

### Claim and ancestry

PASS. The canonical walk is rooted at a retained filesystem-root descriptor,
uses component-relative no-follow opens, and retains the complete canonical
chain through finalization. Exclusive directories are private and bind full
metadata; shared ancestors bind identity and security while allowing unrelated
direct-child churn.

Claim publication uses descriptor-relative Linux
`renameat2(RENAME_NOREPLACE)`. The source name, retained child identity, and
full fingerprint are checked immediately before the syscall. A foreign
destination cannot be replaced: the destination is preserved, the journal is
poisoned, and no claim succeeds.

### Recovery and failure

PASS. A historical tree cannot be repaired into eligibility. Recovery requires
the exact private scaffold, empty private lock, exact current staging-name
grammar, and at least one complete staging containing only private, singly
linked, size-bounded `reservation.json` and `staging.json` leaves. Predecessor
names, `claim.json`, unsafe leaves, missing structure, zero staging, and
incomplete, mutated, or foreign state are preserved and block a claim.

One complete candidate is rehashed and resumed. Authority-equivalent complete
duplicates use the frozen lexical-first policy and are removed only after full
classification. Conflicting authoritative state is preserved and blocks.
Opened recovery descriptors close on invalid, failure, cleanup, and transfer
paths.

Artifact ownership is registered at journal commit before fallible
post-commit checks. Verification and finalization exceptions terminalize while
the retained claim and parent descriptors remain live. Cleanup removes only an
exact registered artifact; changed, missing, linked, replaced, or foreign
artifacts remain invalid. Journal poison can produce only an identity-bound
failure receipt and can never restore success eligibility.

## Destructive syscall boundary

The review accepts the explicit platform boundary in the frozen handoff. Linux
offers no unprivileged inode-conditional `unlinkat` or `rmdirat`. V7 therefore
binds the full source fingerprint at the final userspace boundary and requires
the exact kernel event and post-state after deletion. A same-UID replacement at
that last boundary poisons success, but the kernel cannot guarantee that the
raced replacement object itself is preserved.

This limitation is distinct from canonical claim publication. The
`RENAME_NOREPLACE` claim guarantee preserves a racing foreign destination
absolutely. In every deletion-boundary test, the important safety property
holds: unexpected history permanently prevents a successful result.

## Import and science contract

PASS. Ordinary import exposes no lifecycle function, class, authority object,
partial stage, or writer. The sole production operation exists only inside the
isolated canonical script entry. The synthetic lifecycle is separate and
production-ineligible.

The V7 experiment matches V6 in every value except its fresh output path. It
retains seed `20260710`, N=5, the same five frames, fresh initialization,
AdamW at learning rate and weight decay `1e-4`, 400 updates, batch size 5,
2,000 exposures, float32 without autocast, gradient clipping at `1.0`, four
losses weighted `0.25`, final-update-only selection, and the unchanged
matched/wrong-RGB verification and final gate. The schedule SHA-256 is
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`.

The isolated exact launcher remains GPU0-only on the AMD Radeon AI PRO R9700,
forbids the Raphael iGPU, caps RGB workers at five, and sets one native math
thread per process. No GPU launcher was invoked during review.

## Verification

All pytest commands disabled external plugins, hid accelerators, and capped
native math threads at one.

```text
V7 focused transaction/source/science suite:       102 passed in 5.48s
retained V6 author suite:                            40 passed in 1.63s
applicable retained V1-V5 author closure:           103 passed, 8 deselected
V7 isolated CPU contract smoke:                     PASS
V7 static authority preflight:                      PASS
py_compile for all four frozen V7 artifacts:        PASS
git diff --check for frozen V7 closure:             PASS
canonical V7 experiment output:                     ABSENT
exact execution or optimization:                    NOT RUN
production payload or GPU access:                   NONE
```

The unfiltered retained matrix reproduced `103 passed, 8 failed`. The eight
failures are exactly the documented historical assertions that the already
consumed V5 review or output must still be absent. The explicit applicable run
reproduced `103 passed, 8 deselected`; no security, science, schedule,
lifecycle, or numerical check was excluded.

The isolated smoke reproduced 400 updates, 2,000 full-panel exposures, schedule
hash `62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`,
and synthetic total loss `0.265`. It did not train or open experiment inputs.

## Review record and authority

The canonical machine-readable source review is
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v7_independent_review_2026-07-13.json`.
It is constructed exactly from `expected_source_review_core(...)` for reviewer
`/root/camera_v5_independent/camera_v7_pre_freeze_review` and only the frozen
policy and executor source bindings.

Its canonical content SHA-256 is
`378a0cd61610800ba65eff9a3ac382fa69640b0c50148f5a00a161bba2641def`.
Its newline-terminated file SHA-256 is
`e581739ffdca18a3302d2fef527a43ef9bf31a87f35f4ca2a8e4cc75116d865e`.

This PASS authorizes only one fresh exact V7 N5 full-panel infrastructure
replacement attempt and its bound metric verification and finalization. It
does not authorize a retry, scientific retry, V5/V6 numeric-state use, N16,
another seed, later training, checkpoint use beyond metric verification,
selection, G2, held-out work, calibration change, runtime, hardware,
navigation, production, or promotion.
