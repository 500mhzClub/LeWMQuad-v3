# Longer return trial: mission component tested, integration incomplete

The completed chained native run reached its outbound goal at frame 3062 and
then made physical return progress, but exhausted the 4,000-navigation-step
mission. Its final result SHA-256 is
`163e92d630d92f9f8633cd3823c7cbf282a92c81bde5d3db998e3a5a8c8b1849`.
It had no verified round trip. A longer prospective development trial can test
whether continued execution reaches home; the existing observations do not
establish that it will.

The proposed next allowance is **8,000 navigation steps**, twice the original
allowance. With the unchanged three warmup commands and ten terminal drain
commands, that means at most 8,013 commands, 8,014 observations and 401,400
physics samples, including 750 settling samples. This is a new global mission
budget, not a reset when outbound arrival occurs. No longer native trial has
been launched, queued or admitted.

## Tested mission component

`lewm/extended_return_budget_mission_development.py`, SHA-256
`7054ac4a100d5d86f7ea3537369677b18cb563f0a1067167b90be7054d9883c4`,
privately binds the original observed mission constructor with an 8,000-step
ceiling. Cooperative inheritance preserves the complete measured settling,
quiet-boundary and measured-floor mission methods. The original constructor
still rejects a budget above 4,000; none of its source or globals changed.

`lewm/tests/test_extended_return_budget_mission_development.py`, SHA-256
`ef63505f514c0e10c003c572f6ee7e325bcb5222c3167f6c3f22845c2efd6a68`,
passed **10 tests in 3.02 seconds** in tool session 84971, using the original
deterministic single-thread environment and `pytest -q -p no:cacheprovider`.
The synthetic tests establish:

- Invalid/noninteger/over-limit budgets are rejected, and all original initial
  settling state is retained.
- With the same supplied synthetic positions and commands, every mission
  receipt before frame 4003 matches the original except the declared global
  budget. At frame 4003 the original expires and the successor remains in
  RETURN, without resetting arrivals or observed-state requirements.
- The successor expires at frame 8003 and preserves its terminal receipt on
  later calls.
- A synthetic late return still needs both quiet boundaries and ten quiet
  intervals. Its result is only an observed round-trip candidate, with native
  verification and continuous-speed qualification explicitly false.
- An invalid late position latches a mission failure despite remaining budget.

These tests supply positions; they do not run a visual tracker, model, mapper,
controller or simulation. They prove mission behavior only. The short test run
overlapped the ongoing non-isolated controller timing comparison. All 2,639
sources bound by that comparison were independently rehashed afterward and
remain unchanged. Both new mission files are outside its frozen source roster.

## Additional limits discovered in the current implementation

Changing only the mission allowance would encounter other hard limits. The
following inspected paths require coordinated treatment before native dispatch;
this is an implementation map, not a certified complete source closure.

| Area | Existing bound or dependency | Relevant source |
| --- | --- | --- |
| Measured surface history | 4,096 observations | `lewm/measured_floor_transport_controller_development.py`, `MeasuredFloorTransportMemory.observe` |
| Paired later floor evidence | 4,096 frames | `lewm/later_floor_evidence_development.py`, `LaterFloorEvidence.record_pair` |
| Active body-projected retained floor patches | Literal 4,096-frame ceiling | `lewm/body_projected_floor_geometry_development.py`, `BodyProjectedFloorGeometry.append_patch` |
| Other retained-patch paths | Literal 4,096-frame ceilings | `lewm/frame_cached_floor_geometry_development.py`, `lewm/retained_floor_patch_development.py` |
| Measured floor transport during missingness | Current pose frame must be below 4,096 | `lewm/measured_floor_transport_development.py`, `composition` |
| Native outcome audit | 4,000-step trajectory and arrival-frame ceilings | `lewm/novel_maze_round_trip_evaluation_development.py`, `evaluate` |
| Collector and sensor/replay population | 4,000 steps / 4,014 observations | `scripts/extended_budget_anchored_maze_development.py` and its explicit private bindings |
| New acquisition candidate | Inherits the same 4,014-observation limit | `scripts/single_read_auxiliary_maze_session_development.py` |

The single-pass controller also has exact type checks for fresh memory/map
installation and the optimized footprint selector. A memory subclass cannot
be substituted without checking these dispatch paths:
`lewm/single_pass_body_projected_controller_development.py`,
`lewm/body_projected_tiled_controller_development.py`,
`lewm/fused_scoped_batched_controller_development.py` and their constructor
dependencies. An unintended fallback could preserve outputs while losing the
measured performance improvement.

Extending the transport pose ceiling must remain consistent in the registration
producer, raw-pose reconstruction and all memory/residual/mission consumers.
The native evaluator must remain outside the controller and public packet
dependency graph. Existing measurement-conflict checks, frame continuity,
calibration, contact rules and evidence retention must not be weakened.

## Work required before a longer native trial

1. Complete the active full-history chained/single-pass equivalence comparison
   and authenticate its result before selecting that optimization for adoption.
2. Implement consistent retained-memory, geometry, pose, recording, sensor,
   replay and evaluator bounds for the complete 8,014-observation population.
   Keep all previously recorded evidence intact and preserve bounded failure
   behavior at the new endpoint. A ceiling such as 8,192 retained frames needs
   actual implementation and tests; it is not enabled by this mission module.
3. Verify the extended controller's complete original decision prefix through
   the prospective budget intervention, accounting only for explicitly changed
   budget/implementation declarations. Establish the first changed requested
   command and terminal decision from actual replay, not an inferred outcome.
4. Establish RAM, disk and persistence bounds for the longer population, and
   bind a fresh native protocol and its source/input identities. Do not resume
   the old terminal episode or presume a longer allowance will yield success.
5. Run and fully audit the prospective episode, including actual preintervention
   physical agreement, complete return/retrace evidence and all negative
   outcomes. Reused-layout success would still leave independent-maze,
   matched-baseline, timing and hardware requirements outstanding.
