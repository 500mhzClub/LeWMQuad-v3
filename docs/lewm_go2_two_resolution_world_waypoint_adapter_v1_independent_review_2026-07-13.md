# Two-resolution world-waypoint adapter V1 independent review

Date: 2026-07-13

Verdict: **BLOCK**

The exact reviewed V1 candidate is behaviorally correct at the G3 V2 path to
world-waypoint boundary, but it does not satisfy the required explicit
no-promotion authority contract. The source and focused tests remain
byte-unchanged by this review.

## Frozen candidate identities

- source: `lewm/planning/two_resolution_world_waypoint_adapter_v1.py`
  - SHA-256: `d580fd758b6ac6b14c0576554824f1825ee679400a3c56cc41100657471c51e8`
- focused tests: `lewm/tests/test_two_resolution_world_waypoint_adapter_v1.py`
  - SHA-256: `7710f91ca7596ce1fb467807f86270913ed685725f679705576dccb1c890f291`
- implementation handoff:
  `docs/lewm_go2_two_resolution_world_waypoint_adapter_v1_handoff_2026-07-13.md`
  - SHA-256: `cbef56f309f476f721421fff7e3cd48be2642bd9b48e1b2a770e7b54d489ee78`

All three identities were recomputed before review and the source/test
identities were recomputed again after compilation and testing. They match the
handoff exactly.

## First-principles review

### Exact G3 V2 authority and liveness: PASS

`ConfigurationPathWorldWaypointIssuerV1` requires the exact
`TwoResolutionConfigurationProjectionV2` and
`TwoResolutionConfigurationPlannerV2` types and requires the planner's exact
projection instance. Issuance and validation both call the projection's
exact-live current-snapshot check and the planner's exact-live retained-path
validator. A second planner over a second projection cannot be substituted.

The downstream receipt binds the snapshot identity, both map-frame identities,
memory configuration, physical content, projection source, frozen profile,
both support identities, both revisions, both shapes, and the ordered retained
path receipt. Reprojection makes the old snapshot and path stale.

### Metric conversion: PASS

Each retained configuration cell is converted by the bound configuration map
frame's `cell_center` operation. The tested translated origin is
`(12.37, -8.91)`: configuration cell `(30, 35)` becomes world centre
`(15.42, -5.36)`, and conversion back through the distinct `0.05 m` physical
frame gives physical-grid boundary coordinate `(61, 71)`. This discriminates
against same-index and zero-origin shortcuts.

The three-cell route `(30,35) -> (31,35) -> (32,35)` has exactly two
configuration steps. The receipt records `2.0` steps and `0.20 m`, enforcing
the `0.10 m` configuration-cell cost rather than the `0.05 m` evidence-cell
cost.

### Route safety: PASS

The adapter never accepts a caller-authored route. Before issuance and again
before validation, the frozen G3 V2 planner requires the exact path object it
issued, the current snapshot/frame/revision/support bindings, every route cell
in current configuration `FREE`, four-connectivity, and exact step count.
`UNKNOWN` and `OCCUPIED` route cells therefore reject upstream of waypoint
conversion.

### Tamper, copy, replay, stale, and foreign rejection: PASS

Receipts are capability-bound to one non-copyable issuer, registered by exact
object identity, integrity-checked against a deterministic canonical core,
rebuilt from the current retained path during validation, and optionally
consumed once. Reconstruction, mutation, copy/deepcopy, reuse, stale snapshots,
and foreign projection/planner pairs reject in the focused tests. The receipt
content hash is deterministic because the issuance capability is deliberately
excluded while every controller-facing binding and ordered waypoint is
included.

### Hardware and promotion authority: BLOCK

The receipt correctly provides `hardware_execution_authorized == False` and
serializes `"hardware_execution_authorized": false`. It does **not** provide or
serialize `production_promotion_authorized == False` anywhere. Instead it
serializes `"development_execution_eligible": true` without an adjacent
machine-readable production-promotion denial.

The bound G3 V2 snapshot is non-promotable and the prose handoff grants no
promotion authority, but neither is a substitute for an explicit denial in
the controller-facing waypoint receipt itself. A downstream consumer should
not need to infer this safety boundary through a nested snapshot hash or human
documentation. This fails the review requirement that both hardware and
promotion authority be explicitly false.

Required closure is an additive, versioned successor that exposes and hashes:

- `hardware_execution_authorized == False`;
- `production_promotion_authorized == False`;
- both corresponding serialized fields; and
- a discriminating test asserting both object properties and canonical receipt
  fields remain exactly false.

The V1 source and tests must remain frozen under the identities above.

## Independent verification

All commands ran CPU-only with:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
HIP_VISIBLE_DEVICES=
CUDA_VISIBLE_DEVICES=
ROCR_VISIBLE_DEVICES=
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

Focused and adjacent suites were sharded into independent processes:

```text
python3 -m pytest -q lewm/tests/test_two_resolution_world_waypoint_adapter_v1.py
5 passed in 22.62s

python3 -m pytest -q lewm/tests/test_two_resolution_configuration_projection_v2.py
14 passed in 39.21s

python3 -m pytest -q lewm/tests/test_two_resolution_frontier_viewpoint_v2.py
8 passed in 49.21s
```

The frozen adjacent identities exercised were:

- G3 V2 projection/planner source:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107`;
- G3 V2 projection/planner tests:
  `8e61d29762cac2095d29c5e6341d63cac803c5f118a3eff7e8525b44b4985a3c`;
- G4 V2 frontier/viewpoint source:
  `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82`;
- G4 V2 frontier/viewpoint tests:
  `c50e0d26be068228fe33530d3b2fa42b7520d20d93a0f6e7dc35a6c567ef963e`.

Compilation passed for the reviewed source/tests and both adjacent G3/G4
sources under Python 3.12.

No data, audit result, model, checkpoint, GPU, G2, held-out, V5, runtime, or
promotion input was opened. This review grants no execution or promotion
authority.
