# Two-resolution world-waypoint adapter V1 handoff

Date: 2026-07-13

Status: **additive implementation candidate; different-agent review required**

## Scope

The adapter closes only the deterministic boundary between a live G3 V2
configuration path and controller-facing world coordinates. It does not plan,
smooth, execute, claim, or authorize hardware motion.

An issuer accepts the exact `TwoResolutionConfigurationProjectionV2` and its
exact `TwoResolutionConfigurationPlannerV2`. For every issuance and validation
it requires the current live snapshot and revalidates the exact live retained
path through the planner. Each path cell is converted through the bound
configuration frame to the centre of its `0.10 m` cell. The output receipt
binds both map frames, both revisions, both shapes, the projection source,
physical content, memory config, fixed profile and support identities, ordered
path, world waypoints, cost in configuration steps, and cost in metres.

Receipts are exact-live, non-copyable, tamper-evident, and single-use when
consumed. They explicitly state `development_execution_eligible=true` and
`hardware_execution_authorized=false`.

## Discriminating coverage

- translated origin `(12.37, -8.91)`;
- high configuration index `(30,35)` maps through world metres to physical
  boundary coordinate `(61,71)`, so same-index grid confusion cannot pass;
- three-cell route has two `0.10 m` steps and a `0.20 m` receipt cost;
- reconstruction, mutation, copy/deepcopy, reuse, foreign planner/projection,
  stale snapshot, wrong 2:1 shape, and wrong metric cost reject;
- no G2, held-out, runtime, promotion, model, checkpoint, or GPU input is used.

## Candidate identities

- source:
  `lewm/planning/two_resolution_world_waypoint_adapter_v1.py`, SHA-256
  `d580fd758b6ac6b14c0576554824f1825ee679400a3c56cc41100657471c51e8`;
- focused tests:
  `lewm/tests/test_two_resolution_world_waypoint_adapter_v1.py`, SHA-256
  `7710f91ca7596ce1fb467807f86270913ed685725f679705576dccb1c890f291`.

Focused CPU-only verification passed `5/5` in 22.73 seconds with all native
numeric thread caps set to one and all GPU visibility variables empty. Both
files compile under Python 3.12.

This handoff grants no execution or promotion authority. A different agent
must review the exact candidate bytes and adjacent frozen G3/G4 behavior.
