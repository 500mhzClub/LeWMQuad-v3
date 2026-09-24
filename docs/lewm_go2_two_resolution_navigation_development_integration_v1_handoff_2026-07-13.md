# Two-resolution navigation development integration V1 handoff

Date: 2026-07-13

Status: **PASS for downstream synthetic development composition; production,
hardware, learned-projection, and held-out authority remain unset**

## Result

The previously passed two-resolution APIs now compose into one exact downstream
chain:

1. a current committed G3 V2 snapshot and component;
2. G4 V2 physical-view state, frontier/viewpoint candidates, and one selected
   exact candidate;
3. one exact synthetic runner-owned G5 outcome, two-grid context, and single-use
   positive or negative evidence;
4. reversible target posterior V1;
5. all-hypothesis-safe target router V2;
6. single-use world-waypoint receipt V2;
7. one sealed controller claim-attempt/raw-trace object; and
8. one observer-only canonical claim evaluation with an exact three-key,
   all-zero evaluator-access ledger.

The implementation is
`lewm/planning/two_resolution_navigation_development_integration_v1.py`.
It contains no new geometry, projection, planning, evidence, posterior, routing,
or claim-scoring math. Each operation delegates to an already-passed owner.

## Frozen candidate

- integration source SHA-256:
  `9ba954c191321c629e01cbd8a447a9aff39cf41b35aef26a12f0f7262bd4a0a4`
- focused test SHA-256:
  `9bf16a8cbb685bf07313f0ebb33df47211399198e25b3033ba4552c35a5ddf9c`

Exact consumed API sources:

- G3 V2 projection/planner:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107`
- G4 V2 frontier/viewpoint:
  `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82`
- two-grid G5 evidence:
  `f731b848f6b7ced3b07e11d4f9edca81daa8c66f083f9d503ed069809e38a9a2`
- reversible posterior V1:
  `6d17d06718df355893fa7a6f2f1f735fcf835933178e53c554f4d60181ae96c3`
- target router V2:
  `c8e071d239d1b9894028752fdc090cc2e1be9273f6f9de5a7c7b4d147741b6d2`
- world-waypoint V2:
  `9b710c6f6044bfefd3fd52bcdbb55a52f890b1fdc6c00629029bbf5a670e8fc1`
- canonical raw claim trace:
  `a41f1fa22f5a90503c82db459ccc9520af334173d416bac0b090308d69cc8fb3`
- observer wrapper:
  `1db940a49f01313b23c5d37699796b52da776a3a5c88bf3af1381d7d58103e30`
- canonical claim evaluator:
  `7ea003160ea03da6e989cb76124501b1e7de8571bf8586870b9c8dd7b42f04df`

Any byte change requires a new review record and new hashes.

## Boundary properties

- The constructor accepts only the exact live projection, planner, G4 issuer,
  G4 planner, G5 issuer, posterior memory, router V2, and waypoint V2 owner
  chain. A mixed-owner chain is rejected.
- The public controller input is an exact current
  `TwoResolutionConfigurationSnapshotV2` and exact component. The module does
  not import or know how that snapshot was projected.
- Route and waypoint objects are validated and consumed before the sealed
  controller trace is returned.
- The sealed trace stores original G3/G4/G5/route/waypoint hashes plus canonical
  route, waypoint, controller-attempt, and raw-trace serializations.
- Exact-object identity and the original issuance hash reject copied, forged,
  mutated, stale, and replayed controller traces.
- Production-promotion and hardware-execution denial are hash-bound at the
  route, retained route receipt, waypoint, controller trace, and observer result.
- The G5 semantic target ID and manifest task object ID are separate, explicit
  bindings. For example, `red` and `beacon_red` are not conflated.
- The controller path imports only the raw-trace builder. The observer and
  canonical evaluator are imported lazily after the controller trace is sealed.
- The raw trace requires `evaluator_feedback_to_controller == []`.
- Observer evaluation is single-use. Missing, additional, non-integer, boolean,
  or nonzero access-ledger entries reject before evaluation.
- Observer output contains no controller callback or execution token.
- Production construction and the production entrypoint fail closed.

## Verification

CPU-only focused integration suite:

```text
4 passed in 93.61s
```

It covers the high-nonzero-index happy path, true canonical claim, lazy observer
loading, route/waypoint consumption, copied outcome and controller artifact,
stale G3 snapshot, observer replay, authority mutation, evaluator feedback
mutation, malformed/nonempty ledger, mixed ownership, and production denial.

Adjacent frozen regression suite:

```text
71 passed in 275.04s
```

The adjacent run covered G3 V2, G4 V2, two-grid G5 evidence, reversible target
belief, target router V2, world-waypoint V2, canonical claim trace, and the
observer wrapper. Combined result: **75/75 passed**.

Both runs used CPU only with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, all GPU visibility variables
empty, and pytest plugin autoload disabled. No G2, held-out, hardware,
production, model checkpoint, dataset, or unfinished learned projection was
opened.

## Readiness verdict

**The downstream chain is ready to accept a reviewed learned-projection G3 V2
snapshot without changing downstream planning semantics.** An additive upstream
coordinator may commit the learned native `0.05 m` projection and pass its exact
current snapshot/component into this API.

This is not yet the complete navigation-work readiness PASS. The learned native
projection still needs its own independent PASS, and a later upstream smoke must
replace the synthetic outcome authority with the qualified runner-owned path.
This candidate grants no authority to use production, hardware, G2, or held-out
inputs.
