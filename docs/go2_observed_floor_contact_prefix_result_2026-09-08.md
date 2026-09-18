# Observed floor-contact prefix result

The explicit contact-policy change passed seven focused tests (2.28 s): measured
floor-only contacts at exact nominal foot spheres are allowed with support still
UNKNOWN; all retained returns, unknown/non-floor contacts from either camera,
non-foot contacts and stale-evidence rejection remain intact. One native scope
test passed in 2.02 s. No native command was made by this prefix replay.

Both fixed corrected seed-2026091001 models passed two identical fresh replays
with unchanged model state. Every compared observation, map receipt, model
forecast and nominal first/eight-step check was exact. Every changed surface
check was exactly the declared observed-floor contact transformation.

JEPA first changed the requested command at tick 42, proposing right_arc
`[0.16, 0, -0.45]` instead of the predecessor's first active zero wait. The replay
stopped at that observation (43 frames), before executing the changed command.
Direct reproduced all 57 observations through its recorded tape end without
any command or terminal difference, retaining waits 36–45 and terminal nominal
infeasibility at 46. Neither method had a controller failure.

This supports a prospective test of the declared ground-contact semantics. It
does not verify native recovery, goal arrival, terrain support, JEPA benefit,
independent-maze navigation, real-time execution or hardware safety. The direct
model's nominal wall-clearance failure remains unchanged.

Root: `go2_observed_floor_contact_prefix_v1_attempt_001`.
The run bound 1,307 source files and took 111.8187625859864 seconds after launch.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 2358f9a4294ea549a54585596ba624724dbac569ea9b57f585a726ef43c8fbb2 |
| seed_2026091001_full_jepa_decisions.json | 49c21df0d84d96a3216af1f6bb61a749065d94f82d434dd4c595b4b69eca48e5 |
| seed_2026091001_full_direct_decisions.json | 3f070216388665181644ff483be215fed4593e42ce9b51bc01ba0f1785cd8cbb |
| result.json | 3873716ece87da3a276f5529cf1909e185764be0615b429161eead7eb812d3b5 |
