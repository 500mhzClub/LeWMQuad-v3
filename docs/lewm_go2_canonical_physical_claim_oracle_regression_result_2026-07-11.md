# Go2 canonical physical-claim oracle regression result

Date: 2026-07-11

Status: authoritative development pass for G0 canonical claim accounting and
G1 oracle ceiling. This result does not evaluate learned navigation.

## Frozen Identities

- evaluator binding SHA-256:
  `2de4ff20cff2901ab07b681f042c231f1a1e06f95a77d8c4ae2c20c9e2bb8112`
- implementation manifest SHA-256:
  `f55656eb303a20a1d2fa99813f2a28d84e822e9240e993422974dd416fa0450b`
- implementation source-map SHA-256:
  `f114cb50fe80fd2f026f9a27b727629885168c694b16de681786ef13a4fb9a0b`
- implementation content SHA-256:
  `9d9beb6b0f00999c7519e0c56bc1ee4ed7fbe20411a5efa313b4e7bcb0923d7b`
- finalized result path:
  `.generated/oracle_positive_control/go2_generalization_v4_development/canonical_physical_claim_v1_report.json`
- finalized result file SHA-256:
  `4093461d842d926d4d351d84dec3bd8dff8a828f8730ef3b78c4a11aadfaee03`
- finalized result content SHA-256:
  `1b22227c0a7b8785033dd1c1e6a770a9108cbbda85698ff9dc9dabc5da0c26cc`

The result content hash was independently recomputed from canonical JSON after
removing only `content_sha256`; it matched exactly.

## Authoritative Result

| Gate | Result |
|---|---:|
| scenes finalized | 24/24 |
| oracle raw attempts | 96 |
| oracle evaluations | 96 |
| oracle accepted and credited task objects | 96/96 |
| scenes completing all four objects | 24/24 |
| eligibility raw attempts/evaluations | 96/96 |
| eligibility accepted and credited witnesses | 96/96 |
| rejected or unverifiable oracle events | 0 |
| rejected or unverifiable eligibility events | 0 |
| duplicate credits | 0 |
| stalls | 0 |
| collisions | 0 |
| actual-yaw directional-polygon collision segments | 0 |
| scenes routed through `OnlineBeliefMap` | 24/24 |

`finalization_passed` is true. The finalized aggregate confirms exact oracle
and eligibility task-pair sets, equality of those sets, all claims accepted and
credited, 24 eligible scenes, 24 all-target oracle scenes, and the zero
collision/stall/polygon gate.

Coverage is diagnostic for this privileged positive control: median final
coverage was `0.621811`, minimum final coverage `0.493637`, and median
normalized coverage AUC `0.337149`. Coverage is not part of this G1 claim
ceiling pass.

## Isolation And Resources

The runner used two fixed six-process CPU stages, spawn start mode, one native
numerical thread per worker, preloaded scene manifests, one preloaded verified
directional policy, and development-manifest ordered merge. The policy payload
was read once and the identical in-memory object was used for routing and
eligibility.

The input ledger records zero worker runtime input-file opens and zero prior
comparator, held-out, sealed, G2, label, image, or model-output payload opens.
The evaluator feedback ledger is exactly zero for controller reads, callbacks,
and evaluator-derived termination. GPU0 and the integrated GPU were both unused.

## Consequence

The benchmark can express the required 24-maze, 96-object task without a
planning, following, collision, or claim-accounting ceiling. G0 canonical claim
integrity and G1 privileged oracle ceiling are satisfied for this frozen
development corpus. This does not imply learned perception, memory,
exploration, target conversion, or held-out generalization; those remain later
gates.
