# Density-routed floor calculation: verified registration result

The density-routed candidate completed the original 854-observation tracking
prefix with all complete registration receipts equal to the recorded originals,
all original/candidate registration states equal after every observation, and
all public input fingerprints unchanged. Paired registration time was
63,595.648999 ms original versus 51,912.851570 ms candidate: an 18.370435105055247%
reduction on the shared host. This is a component timing result.

The independent completion audit reconstructed all 854 actual original public
input fingerprints, alternating timing rows and complete timing totals, checked
strict successful comparison flags, verified the execution owner ended, and
rehashed source and raw artifact bindings. It did not independently rerun the
registration calculations. The executing replay performs those numerical
comparisons directly before writing each successful row.

## Preserved cause and negative predecessor

The previous eligible-cell implementation produced identical registration
results but was 8.386756040298728% slower overall. The fixed recorded workload
probe showed sparse primary-camera inputs benefited while all auxiliary-camera
cells were eligible and gathering them was expensive. No ambiguous-boundary
dense fallback occurred in that diagnostic.

The successor samples pixels only to route computation. It chooses the original
dense kernel for estimated occupancy above 50%; otherwise it chooses the
eligible-cell kernel. Both apply full original validation and floor gates. The
candidate retains the original dense fallback for ambiguous numerical boundaries.
No scientific threshold or camera-specific output rule changes.

The fixed successor workload probe compared 96 alternating pairs across frames
0, 100, 400 and 800 and both cameras. Output array bytes remained identical.
Primary-camera timings improved 33.94–41.28%; auxiliary timing stayed within
0.61% of the original. These development observations informed this performance
candidate and are not independent navigation evaluation.

## Exact evidence identities

- Recorded workload probe: `docs/go2_density_routed_floor_index_recorded_workload_probe_2026-09-11.json`,
  SHA-256 `77126309d0d20345484b631f99ca090fe836f4d03574a7135b34fa51d6b36c6e`.
- Registration attempt: `go2_density_routed_floor_registration_prefix_v1_attempt_001`,
  launch `db41edbc9e7d8726dd208080fe85f40dbf5237bf88cd63f3c8752403f92b4139`,
  result `f9d3bb79d485dece8c7490fecbb2233d99e82219365ebc9a710b67dd360ac0b6`.
- Registration execution record: `docs/go2_density_routed_floor_registration_execution_2026-09-11.json`,
  SHA-256 `4aca6e8a962d12beb5efbe6cfc2eb65bb3779cbfe052e0928fb3173f2610680f`.
  Owner PID 2849625, creation time 1789130976.86, ended; tool 13607 exited 0.
- Registration completion audit:
  `docs/go2_density_routed_floor_registration_completion_verification_2026-09-11.json`,
  SHA-256 `278f99ca25615e525a55ae944ed69b461aa2e54e20c395cf3716733ca57de61b`.
  Tool 31602 exited 0.
- Controller preparation: `docs/go2_density_routed_floor_controller_preparation_2026-09-11.json`,
  SHA-256 `c9f26d360f20910538804d82ca907fcc7e4046378cf0e84a60f2d2b3cb36f75b`.
  It binds 2,297 sources; tool 61236 exited 0.

## Complete controller integration

The prepared controller changes the private floor-index binding in registration
and per-observation mapping. It preserves the existing packed memory, visibility
patches, footprint reuse, model, selector, observers and mission. Tests passed:
21 floor-kernel tests, two registration binding tests, 13 completion-evidence
corruption tests, 11 controller integration tests and 16 paired replay tests.
The controller tests include actual public packets, whole registration and map
receipts, robot-footprint equality, immutable cached arrays and cache closure.

The fixed next comparison is `scripts/replay_go2_density_routed_floor_late_history_v1.py`
under its separately recorded protocol. It compares the completed visibility-
batched baseline and this candidate over 1,428 observations, including 1,425
original learned forecasts and seven original retained-state checks. Its source
preflight passed before execution. Starting that process is not a completed
controller result; its output and actual process owner must be checked.

No queued navigation controller has adopted the candidate. Whole-controller
timing, prospective physics, independent-maze round trips, causal comparisons,
the 100 ms control deadline and bounded hardware evidence remain outstanding.
