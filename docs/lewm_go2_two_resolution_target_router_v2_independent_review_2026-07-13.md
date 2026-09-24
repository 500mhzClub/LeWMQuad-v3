# Two-resolution target router V2 independent review

Date: 2026-07-13

Verdict: **PASS**

The frozen V2 remediation closes both blockers in the V1 review at SHA-256
`ec623e98244b66abd78bb12c1350d98aeed04b9775b5232948998d2d5e323c0c`.
It excludes the union of every live posterior hypothesis from terminals and
complete paths, and it rejects consistently rehashed authority or semantic
mutations against router-owned original issuance state.

The candidate source and tests were not edited by this review.

## Frozen identities

- V2 source, `lewm/planning/two_resolution_target_router_v2.py`:
  `c8e071d239d1b9894028752fdc090cc2e1be9273f6f9de5a7c7b4d147741b6d2`;
- V2 tests, `lewm/tests/test_two_resolution_target_router_v2.py`:
  `9b92385a3e9114c2675885a1d4c9be4008706c844572f94872e4f4d141e1ea07`;
- V2 handoff,
  `docs/lewm_go2_two_resolution_target_router_v2_handoff_2026-07-13.md`:
  `aee05104527eec7d3b27b47225146912c62f01a76621fa53977d89912cac9f34`.

All three identities were recomputed before and after the review commands and
match the handoff exactly.

The frozen V1 source/tests also remain unchanged:

- `fbef970bc8637c2c87159edaeffa779b3da12b7f6b9bd4ae67af4f14dd3df252`;
- `506858cdae10bd2ff8b9644a839c2034fd56b1487e2489bdd5ada9c92f52a3b6`.

## V1 counterexamples

### All-mode path exclusion

The exact V1 counterexample was reproduced with two separated posterior modes:

```text
hypotheses  [[(25, 5)], [(10, 5)]]
V1 path     (5,5),(6,5),(7,5),(8,5),(9,5),(10,5),(11,5),(12,5),(13,5)
V1 crossing [(10, 5)]
```

V2 derives one union from the exact current posterior before candidate
generation, combines it with the runner-excluded target cells, and applies the
combined forbidden set to every terminal and every complete retained path.
For the same posterior and start, it issued:

```text
V2 path     (5,5),(5,6),(6,6),(7,6),(8,6),(9,6),(10,6),(11,6),(12,6),(13,6),(14,6)
V2 crossing []
V2 validate accepted_safe_plan
```

The plan serializes the complete union and a hash committed to the exact
posterior snapshot. Validation re-derives the union from the exact current
posterior and requires equality before consumption. Both the terminal and
retained G3 V2 path are therefore all-mode safe.

### Rehashed authority mutation

V1 again accepted a retained receipt after
`production_promotion_authorized=True` and its public canonical checksum were
changed consistently:

```text
V1 rehashed promotion ACCEPTED
```

V2 retains each issued plan's original `content_sha256` in router-owned state
separate from the caller-visible plan. Its plan hash includes the complete
nested retained receipt. Validation also explicitly checks both plan authority
fields and both nested receipt authority fields before accepting the current
route.

Independent probes changed and consistently rehashed each surface:

```text
plan    production_promotion_authorized  stored_equals_rehashed=False  REJECTED
plan    hardware_execution_authorized    stored_equals_rehashed=False  REJECTED
receipt production_promotion_authorized  stored_equals_rehashed=False  REJECTED
receipt hardware_execution_authorized    stored_equals_rehashed=False  REJECTED
receipt initial_heading_error_rad         stored_equals_rehashed=False  REJECTED
```

The authority probes are rejected directly by explicit false checks. The
non-authority semantic probe remains internally checksum-consistent but is
rejected because its recomputed plan identity differs from the separately
stored original issuance identity. Mutating nested path or receipt semantics
cannot preserve both plan integrity and the stored issuance identity.

## Live bindings and lifecycle

V2 requires the exact projection and planner pair, exact live G3 V2 snapshot
and component, exact current posterior snapshot, and exact retained G3 path.
It binds both frames, revisions, supports, projection source, candidate domain,
posterior/evidence chain, selected mode, all-mode union, candidate set, route
cost, start pose, and terminal orientation through the nested receipt and V2
plan commitment.

Independent adversarial probes produced:

```text
copied component    REJECTED: SnapshotBindingError
copied posterior    REJECTED: TwoResolutionTargetMemoryBindingError
consumed replay     REJECTED: TwoResolutionTargetRouteBindingError
stale G3 snapshot   REJECTED: SnapshotBindingError
```

Plans are non-copyable, exact-instance, and consume once. Posterior or G3
advancement invalidates the route before reuse. Production router V2 authority
remains unset.

## Waypoint composition

The V2 route's retained `ConfigurationPathV2` remains the exact live path
issued by the frozen G3 V2 planner. The reviewed composition passes that path
directly into `ConfigurationPathWorldWaypointIssuerV2`, which revalidates it
against the same projection/planner/snapshot, converts every `0.10 m`
configuration cell through the bound world origin, and produces the same goal
cell as the target-route receipt.

Both route and waypoint artifacts explicitly serialize and integrity-check
`production_promotion_authorized=false` and
`hardware_execution_authorized=false`. Waypoint and route receipts each
validate and consume exactly once. No coordinate-grid reinterpretation or
caller-authored path is introduced at the composition boundary.

## Verification matrix

All tests ran in parallel CPU-only shards with native numeric thread caps set
to one and `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, and
`ROCR_VISIBLE_DEVICES` empty:

```text
target router V2                         8 passed in 77.38s
target router V1                         4 passed in 44.65s
two-resolution posterior                 6 passed in 21.41s
two-resolution G5 evidence              22 passed in 74.44s
G3 V2 projection/planner                14 passed in 40.39s
G4 V2 frontier/viewpoint                 8 passed in 51.81s
world-waypoint adapter V2                6 passed in 23.70s
legacy G5 reversible target belief      46 passed in 13.01s
```

Adjacent frozen identities:

- posterior source/tests:
  `6d17d06718df355893fa7a6f2f1f735fcf835933178e53c554f4d60181ae96c3` /
  `6c20fd5b237673aed10a2d03759ed87af057cc55da461a45f046444367540cee`;
- two-grid evidence source/tests:
  `f731b848f6b7ced3b07e11d4f9edca81daa8c66f083f9d503ed069809e38a9a2` /
  `e33dbf595fe27c18c2fddf89cc8f22a005574f67348c2d8746b8ee1ca039de26`;
- G3 V2 source/tests:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` /
  `8e61d29762cac2095d29c5e6341d63cac803c5f118a3eff7e8525b44b4985a3c`;
- G4 V2 source/tests:
  `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82` /
  `c50e0d26be068228fe33530d3b2fa42b7520d20d93a0f6e7dc35a6c567ef963e`;
- world-waypoint V2 source/tests:
  `9b710c6f6044bfefd3fd52bcdbb55a52f890b1fdc6c00629029bbf5a670e8fc1` /
  `3c00554aa14a2a0a98a914e552b7fdb8c4e7cdccbd80fe7b25aeb32e0c2ef440`;
- legacy G5 source/tests:
  `b7f42f90accc9b44f9c38c386318e6775a26d3184d03086d14904487384f14f3` /
  `813ede3e46770b41d617ab90efb5e43ba77c4f99e411c44ce4638f2707cc90ce`.

Compilation passed for the V2 source/tests and adjacent V1, posterior,
two-grid evidence, G3, G4, and waypoint V2 sources under Python 3.12.

No candidate source/test was edited. No navigation rollout, data, audit
result, model, checkpoint, GPU, G2, held-out, runtime, or promotion input was
opened. This PASS covers only the development router remediation and grants no
hardware execution, production promotion, or navigation-performance claim.
