# Go2 categorical radial ladder v2 optimizer amendment

Date preregistered: 2026-07-10 21:48 BST

Status: active; written before any v2 model output

## Scope

This amendment changes exactly one part of the train-only N=1/4/16
categorical-radial ladder: the learning-rate schedule. The frozen data,
selected frame identities, architecture, initialization, loss, batch order,
stage budgets, batch sizes, optimizer family and defaults, weight decay,
gradient clipping, evaluation cadence, controls, and terminal gates remain
unchanged.

This is still an implementation diagnostic. It cannot select or promote a
checkpoint, authorize non-train access, pass G2, or support a generalization
claim.

## Immutable evidence

The amendment responds only to the completed v1 artifact:

- result:
  `.generated/go2_categorical_radial_micro_overfit/v1/seed_20260710_ladder_result.json`;
- result file SHA-256:
  `72e4ecbe6b9e9024bb910e5231deb42e2d73f3187babd2a9af518251cbb7c2a2`;
- result content SHA-256:
  `02c627eb01e42a5b7e8ea57e5bd4bde3d1fc2ca0667abdd9dd1cf8162beacd52`;
- frozen ladder manifest:
  `.generated/go2_categorical_radial_micro_overfit/v1/ladder_manifest.json`;
- manifest file SHA-256:
  `967812399045b29e8be316f2f87bc16f02d681b0ea01884513c6b4f29bbe4b12`.

V1 passed N=1 at its fixed terminal checkpoint. N=4 passed its complete gate
at every 100-step evaluation from step 300 through step 1,400, then failed at
the mandatory step-1,500 checkpoint. From step 1,400 to 1,500, balanced NLL
rose from `0.00088808` to `0.00743824`; OCCUPIED recall fell from `1.0` to
`0.98387`; batch loss rose by 5.42 times; and the pre-clipping gradient norm
rose by 10.05 times. The wrong-view control remained separated by more than
5.0 NLL. Because N=4 is full-batch, this is evidence of a late constant-step
AdamW excursion, not missing four-frame representational capacity.

No step-1,400 weights are accepted, selected, or promoted. V1 remains a
terminal failure under its preregistered rule.

This diagnosis is deliberately bounded. The long passing interval establishes
that the architecture can represent these four frames and is consistent with
a late constant-step AdamW oscillation; one seed and one trace do not uniquely
identify the optimizer as the cause. The terminal OCCUPIED recall failure was
two wrong cells out of 124, while the gate permits one. V2 is therefore an
adaptive, post-hoc schedule choice from a narrow train-only trace, not new
generalization evidence. Exactly this one schedule is frozen before output;
there will be no schedule sweep. ROCm `grid_sample` backward also remains a
warn-only nondeterministic kernel, which every artifact must disclose.

## Sole v2 change

Each stage uses a deterministic cosine learning-rate schedule with no warmup.
For one-indexed update `u` in a stage of `U` updates, set the optimizer learning
rate immediately before `optimizer.step()` to:

```text
lr(u) = 1e-5 + 0.5 * (2e-4 - 1e-5)
                  * (1 + cos(pi * (u - 1) / (U - 1)))
```

Therefore update 1 uses exactly `2e-4` and update `U` uses exactly `1e-5`.
The stage budgets remain `U=1000`, `1500`, and `2000` for N=1, N=4, and N=16
respectively. The learning rate is assigned before every update; it is not
advanced after an update by library-dependent scheduler semantics.

This amendment does not change either separately registered N32 optimizer
branch. It must not be applied silently to N32 or any full-dataset run.

Everything else remains as registered in
`docs/lewm_go2_categorical_radial_microfit_protocol_2026-07-10.md`:

- each stage restarts from the identical seed-specific initial state;
- AdamW betas, epsilon, and all other defaults remain unchanged;
- weight decay remains `1e-4` and gradient clipping remains `1.0`;
- batch sizes remain 1, 4, and 4;
- evaluation remains every 100 updates;
- the complete fixed budget is consumed;
- authoritative execution stops at the first failed terminal stage;
- all fit gates and the deterministic wrong-view control remain unchanged;
- there is no early stopping, EMA, checkpoint averaging, retry, or best-step
  selection.

## Decision rule

The v2 result is judged only at each stage's fixed final checkpoint. If N=1 or
N=4 fails, execution stops and no larger stage is run. If both pass, N=16 is
run under its independently sized cosine schedule and unchanged gate. A v2
N=16 pass licenses only the already-registered N32 fit-panel diagnostic; it
does not license G2 access or promotion.
