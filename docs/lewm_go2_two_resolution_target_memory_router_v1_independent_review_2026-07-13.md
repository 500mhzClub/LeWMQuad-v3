# Two-resolution target memory and router V1 independent review

Date: 2026-07-13

Verdict: **BLOCK**

The frozen posterior implementation passes its transition, normalization,
recovery, evidence-ownership, lattice-binding, replay, and authority checks.
The frozen router passes its ordinary focused tests but admits two adversarial
cases that prevent navigation composition: a route may cross an unselected
live target hypothesis, and a consistently rehashed authority mutation is
accepted by exact-live validation.

The candidate source and tests were not edited by this review.

## Frozen identities

- posterior source,
  `lewm/planning/two_resolution_reversible_target_belief_v1.py`:
  `6d17d06718df355893fa7a6f2f1f735fcf835933178e53c554f4d60181ae96c3`;
- posterior tests,
  `lewm/tests/test_two_resolution_reversible_target_belief_v1.py`:
  `6c20fd5b237673aed10a2d03759ed87af057cc55da461a45f046444367540cee`;
- router source, `lewm/planning/two_resolution_target_router_v1.py`:
  `fbef970bc8637c2c87159edaeffa779b3da12b7f6b9bd4ae67af4f14dd3df252`;
- router tests, `lewm/tests/test_two_resolution_target_router_v1.py`:
  `506858cdae10bd2ff8b9644a839c2034fd56b1487e2489bdd5ada9c92f52a3b6`;
- candidate handoff,
  `docs/lewm_go2_two_resolution_target_memory_router_v1_handoff_2026-07-13.md`:
  `a56346b38133003a6b89f2e2c53c114913e9f9195be19b268edc2935eef0a48f`.

All identities were recomputed before and after the review commands.

## Posterior review

### Transition arithmetic and reversibility: PASS

The first positive update transfers only bounded unlocalized mass. In the
frozen discriminating example it produces configuration-cell masses
`(15,15)=0.2975`, `(16,15)=0.085`, and unlocalized mass `0.6175`, exactly as
derived from `0.50 * 0.85 * 0.90` transfer and the aggregated physical-child
probabilities. Existing modes are not decayed by a positive update.

The following negative update changes only the certified visible cell. Its
mass multiplier is `0.49`, while the unrelated mode is unchanged and removed
mass returns to the unlocalized reservoir. A later positive update increases
the suppressed cell again and total mass remains normalized.

An independent 128-update maximum-confidence stress test produced:

```text
initial                         0.29749999999999993
after_128_negative              1e-15
configured_floor                1e-15
sum(cell_mass)+unlocalized      1.0
later_positive                  0.29750000000000093
later_positive > suppressed     True
```

Negative evidence therefore does not underflow or erase stored support, and a
later positive observation recovers it.

### Two-grid modes and bindings: PASS

Physical evidence is owned on the `0.05 m` lattice and converted through the
shared nonzero world origin into `0.10 m` configuration cells. High physical
indices cannot be reused as configuration indices. Separated four-connected
components remain separate deterministic hypotheses ordered by mass, peak
mass, and peak cell.

The memory accepts only exact evidence owned and consumed by its exact
`TwoResolutionTargetEvidenceIssuerV1`. Evidence, context hash, and raw outcome
replay are rejected. Frames, origins, both shapes, both revisions, projection
source, profile, supports, runner execution identity, checkpoint, calibration,
and exact-simulation taint form the immutable execution binding. Posterior
snapshots are non-copyable, exact-instance objects and become stale after any
later memory revision. Production memory authority remains unset, and the
memory/config/snapshot all explicitly deny production promotion and hardware
execution.

## Router review

### Ordinary deterministic behavior: PASS

For high-index physical evidence around `(50,50)`, the target peak is correctly
converted to configuration cell `(25,25)`, not left at `(50,50)`. Two repeated
issuances are byte-identical. From `(5,5)` the frozen deterministic route ends
at `(17,17)`, contains 24 configuration steps, costs `2.4000000000000004 m`,
and records terminal yaw `0.7853981633974483 rad`. The complete retained path
is current G3 V2 confirmed `FREE`, four-connected, and exact-live.

The normal one-mode case excludes its selected hypothesis and runner-excluded
target cells from the terminal and path, faces the target mean, records the
`0.10 m` path-cost and `0.25 rad` bearing contracts, rejects copied plans,
supports consume-once validation, becomes stale after posterior revision, and
serializes both authority denials as false.

### Finding 1: route crosses an unselected live hypothesis

Severity: **blocking navigation safety defect**

The handoff requires the terminal and complete retained path to exclude every
live posterior hypothesis cell. The implementation constructs candidates and
checks paths against only the hypothesis currently being scored. It does not
form or bind the union of all live hypothesis cells.

The independent collinear counterexample used:

- stronger mode: configuration `(25,5)`, sourced from physical `(50,10)` and
  `(51,10)`;
- weaker separated mode: configuration `(10,5)`, sourced from physical
  `(20,10)`;
- route start: configuration `(5,5)`.

The router selected the stronger mode and issued:

```text
selected hypothesis  [(25, 5)]
goal                 (13, 5)
path                 (5,5),(6,5),(7,5),(8,5),(9,5),(10,5),(11,5),(12,5),(13,5)
unselected crossing  [(10, 5)]
```

The exact router validator also accepts this issued path. It is configuration
`FREE`, but it violates the target-safety contract because `(10,5)` is a
second live target hypothesis. A successor must compute the union of every
current hypothesis, exclude that union from every terminal and complete path,
and bind the exact union into the receipt and validation state.

### Finding 2: rehashed authority mutation validates

Severity: **blocking authority-integrity defect**

The router records only exact plan object identity. It does not retain the
original issued receipt hash outside the mutable object, rebuild the expected
receipt, or explicitly recheck both authority fields during validation.
`receipt.assert_integrity()` therefore proves only that the current fields
match the current unkeyed hash; it does not prove they match issuance.

The independent counterexample:

1. issue a normal exact-live plan;
2. set `plan.receipt.production_promotion_authorized=True` with
   `object.__setattr__`;
3. recompute `content_sha256` from the receipt's canonical current core;
4. call `router.validate` with the original current snapshot, component,
   posterior, and exact plan.

Observed result:

```text
validate_accepted_rehashed_promotion_true True
```

The same structural gap applies to hardware authorization and other receipt
semantics. Frozen dataclasses and unkeyed hashes are not issuance authority.
A successor must retain the original receipt content identity outside the
plan, rebuild and compare the complete expected receipt on every validation,
and explicitly require both plan-level and receipt-level production and
hardware authority values to remain exactly false.

## Verification matrix

All tests ran as independent CPU-only shards with native numeric thread caps
set to one and `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, and
`ROCR_VISIBLE_DEVICES` empty:

```text
two_resolution_reversible_target_belief_v1  6 passed in 20.99s
two_resolution_target_router_v1             4 passed in 42.84s
two_resolution_target_evidence_v1          22 passed in 73.33s
two_resolution_configuration_projection_v2 14 passed in 39.99s
two_resolution_frontier_viewpoint_v2         8 passed in 50.25s
legacy reversible_target_belief             46 passed in 12.71s
```

Adjacent frozen identities:

- two-grid evidence source/tests:
  `f731b848f6b7ced3b07e11d4f9edca81daa8c66f083f9d503ed069809e38a9a2` /
  `e33dbf595fe27c18c2fddf89cc8f22a005574f67348c2d8746b8ee1ca039de26`;
- G3 V2 source/tests:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` /
  `8e61d29762cac2095d29c5e6341d63cac803c5f118a3eff7e8525b44b4985a3c`;
- G4 V2 source/tests:
  `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82` /
  `c50e0d26be068228fe33530d3b2fa42b7520d20d93a0f6e7dc35a6c567ef963e`;
- legacy G5 source/tests:
  `b7f42f90accc9b44f9c38c386318e6775a26d3184d03086d14904487384f14f3` /
  `813ede3e46770b41d617ab90efb5e43ba77c4f99e411c44ce4638f2707cc90ce`.

Compilation passed for both candidate sources/tests and the adjacent two-grid
evidence, G3, and G4 sources under Python 3.12.

No candidate source/test was edited. No navigation rollout, data, audit
result, model, checkpoint, GPU, G2, held-out, runtime, or promotion input was
opened. This review grants no execution or promotion authority.
