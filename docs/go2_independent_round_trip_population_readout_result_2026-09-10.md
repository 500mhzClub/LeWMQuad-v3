# Paired population completion/readout and original maze3 completion

Implemented fixed within-layout startup references and complete 32-case paired
readouts. All eight layouts remain in each arm's denominator. Audited early
physical/acquisition stops carry unavailable-startup receipts instead of being
discarded or labeled matched. Readouts retain failure reasons, native/observed
arrivals, distinct versus repeated crossings, contacts, strict visibility and
complete decision timings. Success claims are checked against the original
joint outcome and its physical/mission consistency conditions.

Verification:
`go2_independent_round_trip_population_readout_verification_2026-09-10.json`,
SHA-256 `7d264f828f5be0a6f40b3e7cf2d500b17a48e5fad3e23cb1162fa51297b0d08a`.
Protocol: `go2_independent_round_trip_population_readout_v1_2026-09-10.md`.

Tests: handle 45148, exit 0, 39 passed in 1.99s. Source/artifact verification:
handle 17144, exit 0, 1,953 source bindings checked before and after. The original
queued sources and fixed comparison manifest remain unchanged. No new-layout
native scene, neural inference or model training was performed in these checks.
The population tests use synthetic accounting fixtures and do not establish
navigation reliability or planning/memory advantage.

Original episode31 is now top-level complete:

- Root: `go2_recent_qualified_direct_flow_maze03_pilot_v1_attempt_001` under the
  fixed development artifact BASE.
- Result SHA-256:
  `330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723`.
- Original handle 25801 exited 0. Its final output reported successful complete
  original input verifiers with fresh before/after scoped-file hashing.
- This turn separately checked all 1,843 source bindings and 16,870 result-bound
  artifacts, then admitted the original worker/raw audit/prefix relationship.
  It did not rerun the full original input verifiers or raw model replay.
- Raw sensor/model-command/command auditing, strict visibility and the fixed
  intervention prefix pass. No hard-measurement failures were reported.

The negative outcome remains explicit: 2,805 observations/decisions, 2,804
completed command intervals, 140,950 physics samples and ten terminal zero
commands. No observed or native arrivals, cell crossings, distinct edges,
return traversal or contact samples occurred. The schedule terminal is
`SENSOR_OR_MODEL_FAILURE`; there was no physical or acquisition stop. The
previous complete-stream scan identified floor-registration correction-gate
failure beginning at observation 2794. Contact-free collection and passing
sensor audits do not establish useful navigation.

The new readout was exercised on this actual completed original episode:
median observation/control 1,298.419925ms; p95 1,686.6427628ms; maximum
2,410.366832ms. All 2,805 decisions exceeded the 100ms command interval.
Median iteration including receipt persistence was 1,348.002878ms. Physics
was paused during computation; this is not real-time or hardware validation.
Original top-level wall time was 8,882.640957244905s.

The original waiter 2641948 accepted this final result and the already completed
correction result, then launched child 2653445 with the exact two result hashes.
Both remained live at the final process check. The child was running full input
admission; the six-case scene output root was not yet present. This is the
original authorized queue, not a duplicate or retry.

Status: 31 completed top-level audited native episodes, zero verified round
trips. Independent-population input admission and launcher are still required.
Inspect the queued expanded-model outcomes before freezing or executing that
population; correct implementation failures on existing development evidence.
Reliable unseen-layout round trips, useful physical backtracking, realistic
timing and bounded real-platform evidence remain outstanding.
