# Paired late-history reuse replay prepared

The harness compares the existing frozen-footprint controller with selection-
local exact footprint reuse, using two independently stored copies of the
original assigned model. It rebuilds all 1,428 observations and 1,425 forecasts,
compares complete decisions against the original tape and completed profiler,
and checks retained observed state at seven fixed checkpoints through frame
1427. Timings alternate execution order and measure the incremental candidate
against the frozen-footprint baseline. The original visibility failure at frame
1173 remains explicit.

The final focused run passed 58 tests in 5.50 seconds (session 16051, exit zero):
39 harness cases and 19 reuse implementation cases. These are synthetic replay,
negative-admission, ownership and actual-geometry component tests. They include
shared model/storage, changed weights/gradients, inputs, decisions, command
endpoints and retained state; incomplete reference profiles; live/reused
process identities; invalid timings; and resource admission.

Actual source preflight passed with 2,145 bound sources (session 70415, exit
zero). Available hardware met the prospective 64 GiB RAM, 41 GiB disk and four
physical CPU requirements. A direct process check confirmed that the original
profiler was still live, and the runtime owner gate correctly rejected starting
this replay. The new artifact root remains absent. No trained-model replay or
native scene was started.

Preparation record:
`go2_scoped_footprint_late_history_replay_preparation_2026-09-10.json`, SHA-256
`139369a85a4d025953eafa05bf7cdba003b496530f679175465ef5973505063b`.

Next, review the completed original late-history profile and helper costs,
authenticate its exact result, and run this paired replay if the evidence
supports that expenditure. Source checks establish no speedup, useful navigation
or deployment readiness. The existing native queue remains unchanged.
