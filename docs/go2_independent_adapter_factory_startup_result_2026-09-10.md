# Four-arm adapter factory: completed runtime compatibility check

The separately versioned independent-study factory now loads the byte-identical
planner adapter. Its shared collector and full raw audit both call that factory.
Their complete function bodies remain structurally identical to the earlier
multiarm implementation; the factory rejects incompatible wrapper types and
training mode while retaining all fixed arm/model/controller assignments.

New files:

- `scripts/independent_round_trip_adapter_controller_factory_development.py`
- `scripts/independent_round_trip_adapter_multiarm_episode_development.py`
- `scripts/independent_round_trip_adapter_multiarm_audit_development.py`

All four actual factory/controller paths passed the four-observation startup
check. Persistent JEPA and current-pair JEPA request the left arc; persistent
supervised rollout requests the left turn. All three original factories reproduce
the wrapper-interface failure at observation 3, while the corresponding adapted
factory reaches the planner with no terminal failure. Every expanded forward
output tensor and the actual selected forecast bank match direct original-model
inference exactly. The reactive arm remains model-free, with all four complete
old/new decisions identical and a forward request at observation 3.

Both factories use the same fixed study public mission for layout index 0, but
receive raw packets from the already completed development maze2 startup. This
does instantiate a study public goal; it does not render, observe or navigate an
independent maze, reconstruct the original native mission's decisions, or
establish a valid independent-layout episode. All three actual zero warmup
commands are respected. No packet after the first planning request is consumed,
and no new command is executed. Observed map and retained contact state remain
identical between each paired factory/controller execution.

Result: `cd0af02bdbe978e714e55c1c0253f1fa01df88c4554eeea887deae84e40ae6db`.
Launch: `2cbd89c0be82ddb26957393c9c299c62b1442c59dbfb8284fdabf1ae62aac9d6`.
Output: `go2_independent_adapter_factory_startup_v1_attempt_001` under the external
navigation artifact root. Session 18985 exited zero; wall time after launch was
56.039 seconds. The result binds 1,991 sources and five outputs; its launch binds
741 original artifacts including the completed original failure result.

Focused tests: session 55074 exited zero, 12 passed in 2.62 seconds. Independent
check: session 3635 exited zero, validating all source/output/input bindings,
all four streams and 16 packet-use fingerprints, saved old/new observed-state
comparisons, unchanged reactive decisions and each boundary command. That check
did not rerun neural inference. All 1,929 queued frontier source bindings remained
unchanged. Earlier incompatible source versions and their verification records
remain preserved.

The 32-episode/eight-layout population is still unexecuted. Its eventual launcher
must use the adapter versions, authenticate all completed development results,
and freeze the final controller and comparison protocol before independent
execution. If development evidence requires further controller changes, update
the prospective arm definitions before consuming new-layout sensor data. This
compatibility result establishes no navigation, JEPA, RGB, planning or memory
advantage. The active adapter batch and queued frontier physics remain separate.
