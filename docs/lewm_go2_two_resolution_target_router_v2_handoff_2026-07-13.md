# Two-resolution target router V2 handoff

Date: 2026-07-13

Status: **hash-frozen remediation candidate awaiting independent review**

## Reason for successor

The frozen V1 posterior passed independent review, but its router was blocked
at review SHA-256
`ec623e98244b66abd78bb12c1350d98aeed04b9775b5232948998d2d5e323c0c`.
Two exact counterexamples were reproduced:

1. while targeting a stronger mode at configuration cell `(25,5)`, its path
   `(5,5)..(13,5)` crossed a weaker live mode at `(10,5)`;
2. a caller could change a receipt authority field to true, recompute the
   public unkeyed checksum, and pass V1 validation.

V1 source and test bytes remain unchanged. V2 is an additive policy and
issuance wrapper over the exact retained G3 V2 path and V1 receipt container.

## Frozen identities

- V2 source, `lewm/planning/two_resolution_target_router_v2.py`:
  `c8e071d239d1b9894028752fdc090cc2e1be9273f6f9de5a7c7b4d147741b6d2`;
- V2 tests, `lewm/tests/test_two_resolution_target_router_v2.py`:
  `9b92385a3e9114c2675885a1d4c9be4008706c844572f94872e4f4d141e1ea07`;
- frozen V1 source:
  `fbef970bc8637c2c87159edaeffa779b3da12b7f6b9bd4ae67af4f14dd3df252`;
- frozen V1 tests:
  `506858cdae10bd2ff8b9644a839c2034fd56b1487e2489bdd5ada9c92f52a3b6`.

## Remediation

V2 computes the union of every hypothesis returned by the exact current
posterior snapshot. That union is:

- excluded from every terminal candidate;
- excluded from every complete retained path;
- serialized explicitly;
- committed with the exact posterior snapshot hash;
- re-derived from the exact current posterior during validation.

The exact reviewer counterexample now retains a path that contains neither
`(25,5)` nor `(10,5)`.

V2 also stores the original issued plan content hash in router-owned state.
Validation requires:

- exact plan object identity;
- current posterior/snapshot/component and exact G3 path identity;
- unchanged original issuance content, independently of the object-carried
  checksum;
- both plan authority fields exactly false;
- both retained receipt authority fields exactly false;
- unchanged router-config, posterior, all-hypothesis, path, and receipt
  bindings.

Thus changing a semantic or authority field and recomputing the receipt and
plan checksums still rejects.

## Verification completed by author

- V2 focused suite: `8/8` passed in `72.11 s` under CPU-only one-thread caps;
- the exact unselected-mode crossing counterexample passes;
- four independently parameterized rehashed authority-tamper probes reject;
- a separately rehashed semantic-field probe rejects against stored issuance;
- exact single-use route validation and passed world-waypoint V2 composition
  pass;
- Python compilation, `pyflakes`, and `git diff --check` pass;
- no GPU, G2, held-out, sealed input, or navigation rollout was opened.

## Required independent review

The reviewer must reproduce both V1 counterexamples against V1, then prove
they reject under V2. It must also mutate and consistently rehash each plan and
receipt authority field plus at least one non-authority semantic field, verify
the stored issuance hash is independent, inspect complete-path all-mode
exclusion, and rerun the posterior, G3 V2, G4 V2, G5 evidence, target-router
V1/V2, and world-waypoint V2 adjacent suites.

This candidate is development-only. It grants neither production promotion
nor hardware execution.

