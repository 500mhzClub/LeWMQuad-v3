# Two-resolution world-waypoint adapter V2 handoff

Date: 2026-07-13

Status: **additive implementation candidate; different-agent review required**

## Scope

V2 closes the authority-contract blocker recorded by the independent V1
review while preserving V1 bytes. It converts only an exact live path from the
frozen G3 V2 projection/planner pair into ordered world-frame configuration
cell centres. It does not plan, smooth, execute, claim, authorize hardware, or
authorize promotion.

The typed receipt now contains immutable
`hardware_execution_authorized=False` and
`production_promotion_authorized=False` fields. Both fields are included in
the canonical receipt core and content hash, emitted by `to_dict`, and checked
explicitly by `assert_integrity`. Mutating either field makes issuer validation
fail before the receipt can be consumed. The receipt continues to state
`development_execution_eligible=True`; that development-only eligibility is
therefore accompanied by two explicit machine-readable authority denials.

## Preserved behavior

- the issuer accepts only the exact `TwoResolutionConfigurationProjectionV2`
  and its exact `TwoResolutionConfigurationPlannerV2` instance;
- issuance and validation revalidate the exact live current snapshot and the
  exact live retained G3 V2 path;
- every path cell must remain current configuration `FREE`, four-connected,
  and correctly costed by the frozen planner;
- the receipt binds both map frames, both revisions, both shapes, memory
  configuration, physical content, projection source, fixed profile, both
  support identities, ordered path, translated origin, ordered world
  waypoints, and metric cost;
- configuration cells use the bound `0.10 m` configuration frame, while the
  adjacent physical lattice remains `0.05 m`;
- receipts and issuers remain non-copyable, receipts remain exact-live and
  tamper-evident, and `consume=True` remains single-use.

The discriminating translated-origin case maps configuration cell `(30,35)`
to world centre `(15.42,-5.36)` and physical-grid boundary coordinate
`(61,71)`. Its three-cell route has two `0.10 m` steps and costs `0.20 m`.

## Frozen candidate identities

- source: `lewm/planning/two_resolution_world_waypoint_adapter_v2.py`
  - SHA-256: `9b710c6f6044bfefd3fd52bcdbb55a52f890b1fdc6c00629029bbf5a670e8fc1`
- focused tests: `lewm/tests/test_two_resolution_world_waypoint_adapter_v2.py`
  - SHA-256: `3c00554aa14a2a0a98a914e552b7fdb8c4e7cdccbd80fe7b25aeb32e0c2ef440`

The frozen V1 identities remain unchanged:

- source:
  `d580fd758b6ac6b14c0576554824f1825ee679400a3c56cc41100657471c51e8`;
- focused tests:
  `7710f91ca7596ce1fb467807f86270913ed685725f679705576dccb1c890f291`;
- handoff:
  `cbef56f309f476f721421fff7e3cd48be2642bd9b48e1b2a770e7b54d489ee78`.

## Implementation verification

Focused CPU-only verification passed `6/6` in 21.89 seconds with all native
numeric thread caps set to one and all GPU visibility variables empty. The
source and tests compile under Python 3.12.

Coverage includes both object and serialized authority fields, independent
tampering of each denial field, exact translated-origin geometry, high-index
two-grid discrimination, `0.10 m` route cost, reconstruction, copy/deepcopy,
single-use consumption, foreign projection/planner rejection, stale snapshot
rejection, wrong 2:1 ratio, and wrong metric cost.

No navigation, data, audit result, model, checkpoint, GPU, G2, held-out, V5,
runtime, or promotion input was opened. This handoff grants no execution or
promotion authority. A different agent must review the exact V2 bytes and
rerun focused plus frozen adjacent G3/G4 behavior before composition work.
