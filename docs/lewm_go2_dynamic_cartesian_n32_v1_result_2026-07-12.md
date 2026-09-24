# Dynamic-Cartesian N32 v1 result

Date: 2026-07-12  
Status: complete, finalized, fit gate failed on both registered seeds

## Decision

The dynamic projective cell-square head did not qualify for shared-JEPA
construction. Both registered seeds exhausted the production-faithful and
ceiling-optimizer budgets without a single complete fit-gate pass. Neither
same-scene nor cross-scene holdout payloads were opened, so this result makes no
generalization claim and does not consume G2.

## Immutable evidence

- implementation manifest file SHA-256:
  `f4db2563c0ec22815c0cb831e463b53cc0ebb023ffdc6ab0ef29057e1c34978d`
- implementation manifest content SHA-256:
  `83a6016cf6b11738d847831207cade5ba7a5633893136da8f87e13c0bd667b90`
- seed `20260710` result file/content SHA-256:
  `d35f264d60e18cfa6a8cb8e71ae189ea933b57944a4a71c950e186eaed2e0a41` /
  `a57781edc792115cec4a52ba5dbcb50a06eca1a40eafb17985965fdf4e7f7910`
- seed `20260710` attempt-marker SHA-256:
  `7675e2ba9b98291634d76ba72c75d53505d1236ec9dadf53dcf91e9ed74a7743`
- seed `20260711` result file/content SHA-256:
  `7815b1d749ef54e817bb73c377b60de545f4a67f21ffc93024a6b633f1f166ab` /
  `527aef09ffc69dcea7277aa9143e6326acffc2c93c87ab2ef5529ef59e6d573d`
- seed `20260711` attempt-marker SHA-256:
  `97813188238fa72a7c4a9ddf35c6045459cd364d91b1a92c08d05c82a24aefb9`

Both torch-free finalizer invocations passed. The seed-pair decision is
`shared_jepa_construction_licensed=false`, `g2_licensed=false`, and
`runtime_licensed=false`.

## Ceiling endpoint

| Seed | Hierarchical NLL | UK balanced acc. | FO balanced acc. | Unknown recall | Free recall | Occupied recall | Wrong-scene NLL delta | Wrong-view NLL delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260710 | 0.1107 | 0.9591 | 0.9548 | 0.9424 | 0.9519 | 0.8401 | 1.9241 | 1.1191 |
| 20260711 | 0.1018 | 0.9599 | 0.9627 | 0.9320 | 0.9680 | 0.8547 | 2.0739 | 1.1700 |

Required values were NLL at most `0.03`, both balanced accuracies at least
`0.99`, and all class recalls at least `0.98`, with the distance and
counterfactual gates also required. The counterfactual margins passed by a
large amount: the model was image-conditioned. The absolute map accuracy did
not pass.

Open/rough scenes retained low occupied recall, while enclosed-maze families
retained low free recall. The ceiling curves plateaued near NLL `0.10` with
class tradeoffs across successive evaluations; this is not explained by one
bad initialization or a missing image signal.

## Consequence

Do not widen this result into a held-out claim and do not build the shared JEPA
from this head. The next licensed work is development-only diagnosis:

1. prove the loss, labels, schedule, and gate are attainable with a direct
   per-frame logit upper bound;
2. test an unconstrained image-conditioned dense decoder to separate visual
   encoding capacity from the projective lift;
3. only then redesign the geometric head, preserving the audited attitude and
   projection contracts.
