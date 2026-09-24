# Two-resolution world-waypoint adapter V2 independent review

Date: 2026-07-13

Verdict: **PASS for additive development composition only**

The frozen V2 candidate closes the sole blocker from the V1 independent
review. No blocking defect was found in exact G3 path binding, cross-grid
geometry, one-use behavior, or the two explicit authority denials. This verdict
does not authorize hardware execution or production promotion.

The reviewed source, focused tests, and implementation handoff were not edited
by this review.

## Frozen reviewed identities

- V2 source: `lewm/planning/two_resolution_world_waypoint_adapter_v2.py`
  - SHA-256:
    `9b710c6f6044bfefd3fd52bcdbb55a52f890b1fdc6c00629029bbf5a670e8fc1`
- V2 focused tests:
  `lewm/tests/test_two_resolution_world_waypoint_adapter_v2.py`
  - SHA-256:
    `3c00554aa14a2a0a98a914e552b7fdb8c4e7cdccbd80fe7b25aeb32e0c2ef440`
- V2 implementation handoff:
  `docs/lewm_go2_two_resolution_world_waypoint_adapter_v2_handoff_2026-07-13.md`
  - SHA-256:
    `3794cdbf78f610a794f73ea260b467eaec6e148f8d8e746c60508c81fb2a44eb`

All three identities were recomputed before source review and after independent
testing. They match the handoff exactly.

## Review findings

No blocking findings.

### Exact G3 path binding: PASS

The issuer accepts only the exact G3 V2 projection and planner types and checks
that the planner owns the same projection object. Both `issue` and `validate`
first call the projection's exact-current-snapshot validator and the planner's
exact-live-path validator.

The upstream planner requires the path object to be the exact object registered
by that planner. It also rechecks snapshot hash, both frame hashes, both
revisions, both support hashes, current configuration FREE membership,
four-connectivity, and exact step cost. A caller-authored, copied, foreign, or
stale path therefore rejects before waypoint conversion.

The receipt additionally binds snapshot, memory configuration, physical
content, projection source, profile, both supports, both revisions, both
shapes, configuration origin, retained ordered-path receipt, ordered
waypoints, metric cost, and exact-simulation taint.

### 0.10 m conversion and high-index discrimination: PASS

Every waypoint is created with the exact bound configuration frame's
`cell_center` method. No physical-grid index or zero-origin shortcut appears in
the conversion.

The discriminating translated-origin test uses configuration cell `(30, 35)`.
It becomes world centre `(15.42, -5.36)` on the 0.10 m configuration lattice,
while its location in the adjacent 0.05 m physical grid is `(61, 71)`. The
indices are deliberately unequal, so the test detects same-index confusion.
The three-cell configuration path has two steps and the receipt records both
`2.0` configuration steps and `0.20 m`, not `0.10 m` from a mistaken 0.05 m
scale.

The receipt independently enforces an exact 2:1 physical-to-configuration
shape ratio and recomputes metric cost from the number of ordered waypoints.

### Exact-object and one-use behavior: PASS

Receipts carry an issuer capability outside the canonical content, are retained
in the issuer by exact object identity, and are non-copyable and non-deepcopyable.
Validation rejects a reconstructed receipt even when it has the same canonical
hash. `consume=True` records the exact receipt identity, and every later
validation rejects it as already consumed. Issuers are also non-copyable.

Before consuming, validation repeats current snapshot/path validation,
integrity validation, exact capability/object checks, and a deterministic
rebuild from the current retained path. A failed integrity or binding check
therefore cannot consume an invalid receipt.

### Hashed hardware and promotion denials: PASS

`hardware_execution_authorized` and `production_promotion_authorized` are both
typed `init=False` fields. Construction resets each field to the exact boolean
`False` before hashing. The canonical receipt core and `to_dict` contain both
fields, so both denials are controller-visible and hash-bound.

`assert_integrity` separately requires each field to be the exact boolean
`False` before checking the canonical hash. The focused test mutates each field
independently and confirms validation rejects it. This closes the V1 review
blocker: downstream code no longer needs to infer no-promotion authority from
a nested snapshot or prose.

The literal `development_execution_eligible=True` remains hash-bound but is
strictly adjacent to both explicit denials. It cannot be interpreted as
hardware or promotion authority under this receipt schema.

### V1 preservation: PASS

The V1 source, tests, and handoff remain byte-identical:

- V1 source:
  `d580fd758b6ac6b14c0576554824f1825ee679400a3c56cc41100657471c51e8`
- V1 tests:
  `7710f91ca7596ce1fb467807f86270913ed685725f679705576dccb1c890f291`
- V1 handoff:
  `cbef56f309f476f721421fff7e3cd48be2642bd9b48e1b2a770e7b54d489ee78`

The reviewed V1-to-V2 source diff changes versioned names and adds the two
forced, serialized, hash-bound authority fields plus their explicit integrity
check. Geometry, exact-path validation, cost conversion, issuance, and
consumption behavior are preserved.

## Independent verification

Tests were sharded across CPU processes with native numeric threads capped at
one and GPU visibility disabled:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
HIP_VISIBLE_DEVICES=
CUDA_VISIBLE_DEVICES=
ROCR_VISIBLE_DEVICES=
```

Results:

```text
V2 + preserved V1 waypoint adapters: 11 passed in 45.01 s
G3 V2 projection/planner:            14 passed in 39.14 s
G4 V2 frontier/viewpoint:             8 passed in 49.77 s
Total:                               33 passed
```

Python compilation and `pyflakes` checks passed for the reviewed V2 source and
tests. Compilation also passed for V1 and the adjacent G3/G4 sources.

Adjacent frozen identities remained unchanged:

- G3 V2 source:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107`
- G3 V2 tests:
  `8e61d29762cac2095d29c5e6341d63cac803c5f118a3eff7e8525b44b4985a3c`
- G4 V2 source:
  `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82`
- G4 V2 tests:
  `c50e0d26be068228fe33530d3b2fa42b7520d20d93a0f6e7dc35a6c567ef963e`

No navigation run, data, model, checkpoint, held-out scene, simulator, GPU, G2,
V5, runtime execution, or promotion input was opened. This review approves
only the frozen additive V2 receipt boundary for development composition.
