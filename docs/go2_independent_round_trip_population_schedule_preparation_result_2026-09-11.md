# Bounded fixed population schedule prepared

Implemented `scripts/independent_round_trip_population_schedule_development.py`.
Its state machine allows one collection and one audit with at most two
unaccepted cases, fixes all 32 case dispatch and acceptance positions, and
requires each layout's first-arm parent acceptance before collecting another
arm. Pending audits take precedence over new collections.

Failures stop new dispatch. The other already-running worker can finish and
preserve its evidence; a collection completed after audit failure remains
pending and is not audited or retried. Admission failure has the same draining
behavior. Scientific failure does not replace an arm or layout.

**85 tests passed in 2.35 seconds**, session 6023, exit 0. Four complete virtual
populations exercise differing stage durations. All 64 worker failure
positions are checked for fixed acceptance order, no further dispatch and
drained original in-flight work. Other tests cover the first-arm barrier,
backlog bound, changed completion identities, resource thresholds and state
aliasing. These are scheduling tests; no process or simulator executes.

Preparation verified **2,029 source bindings**, session 31509, exit 0:
`docs/go2_independent_round_trip_population_schedule_preparation_2026-09-11.json`,
SHA-256 `23b91e4fec1d1d3ec9b7fbd77ad9cc916d4f95018a09c17355804f20d687f465`.
All predecessor bindings remain unchanged.

The state machine is not yet a process driver. Next work is connecting actual
spawn workers, start barriers and the frozen collection/audit lifecycle
helpers, with role-aware single-scene admission. CPU-only audit qualification
and measured overlap resource use remain outstanding. Final policy review
and joined queue/input admission still precede independent study execution.

During preparation, the original case-5 audit 2743870 (creation
1789071424.56) and scoped replay 2754886 (creation 1789077365.71) remained live.
The replay had completed all frame comparisons and had no final result or
failure yet. About 583 GiB remained free on artifact storage. No new native
job or full controller replay was started.
