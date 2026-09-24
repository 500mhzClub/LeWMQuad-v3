# Measured settling implementation and mission-prefix result

This successor addresses the ninth pilot's arrival-window braking failure. It
does not fix its separate visual tracking failure or strict visibility issue.
All original sources and failed attempts remain unchanged.

## Implementation and focused verification

MeasuredSettlingRoundTripMission requires consecutive admitted visual3D
positions and at most0.05m/s interval-average displacement before counting a
quiet interval. The original4cm proximity, zero actual previous request,
ten-interval duration, return phase and shared budget remain. Invalid positions
or clocks latch a hold. MeasuredSettlingRoundTripController inherits the
later-floor-resolution stack; its copied advance method differs from the
registered-floor predecessor only by passing p rather than p[:2] to the mission.
Eleven tests passed3.70s in session33711, including the exact AST derivative,
braking, vertical motion, reset, physical-return sequencing, shared budget,
bad input and two public-packet controller integration observations.

The separately named SettledBoundaryRoundTripMission tightens the start of that
window: the first low-motion observation establishes a boundary; only subsequent
intervals with low measured motion at both boundaries count. This avoids counting
time before a quiet boundary was observed. Six tests passed1.83s in51996.
SettledBoundaryRoundTripController uses that mission and retains the original
overlap-retention observer, later-floor map/contact evidence, selector and
residual memory. The failed subpixel candidate is excluded.

Neither rule bounds instantaneous speed between images or calibrates visual
pose uncertainty. All actual2ms native speed, sensor, command, visibility and
physical-return requirements remain unchanged.

## Saved-observation mission comparison

Initial attempt32534 CLOSED exit1 before frame0. Root
go2_measured_settling_mission_prefix_v1_attempt_001;
launchc664b57a3dbd752a36362a9f6e29ba45d41b2222ef7b05bfe0ac260e5430a016;
failure25421b3950be88c00340b586399ec151324bb771509782de6c809c7567637ca2.
Its empty mission stream SHA-256 is
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.
Cause: serialized JSON episode identities are lists; the strict live accessor
requires tuples. No scientific candidate outcome was obtained in that attempt.

A separate adapter reuses the existing readout json_identity helper for exactly
the registered and original visual episode identities, before the unchanged
registered-pose validator. Six tests passed2.50s in12091, covering nonmutation,
malformed/boolean/negative/wrong identities and tampered-pose rejection.

Corrected adapter comparison84035 CLOSED exit0. Root
go2_measured_settling_mission_json_prefix_v1_attempt_001;
resultbf875306754c620d393d7a05496081c318c17d624726d2ae8bf348bab9caa053;
launcheaf8d78e0ee92f58d1e14bda54209229f977a67cef6d6b5db52124154522b147;
stream4c27959b0e9c4c44d4ad24c53351e10ac0a0dbfabb43b8923da95a2454fee4c6;
1549 sources. Resource admission:74,494,914,560 RAM bytes available and
118,645,473,280 artifact-free bytes,3.6% aggregate CPU, one CPU replay beside
the existing native audit. All source/input bindings revalidated after replay.

All1867 original mission receipts and current registered-pose witness checks
passed through frame1866. First counter difference1857: observed interval
speed0.10811437970540624m/s, so the successor rejects this braking interval.
At1866 the original claims arrival/switches RETURN, while the first measured
candidate remains OUTBOUND with9 quiet intervals and a zero-command hold.
The comparison stops there before reading later decisions. This is mission-only
saved-observation evidence, not a raw-controller replay or new physical outcome.
The boundary successor described above has a stricter dwell-start rule and
is evaluated separately by the full-controller replay.

## Full-controller replay is running

Prepared a comparison that rejects every complete decision difference outside
the declared settling/counter/mission fields. Seven tests passed0.10s, including
rejection of changed observer evidence, map state, model forecasts, commands
and earlier candidate arrivals.

Preflight45491 CLOSED exit0:1555 sources,74,205,945,856 available RAM bytes,
118,631,505,920 artifact-free bytes;8GiB RAM and1GiB output above40GiB reserve
admission passed. Full replay56186 is LIVE:
go2_settled_boundary_controller_prefix_v1_attempt_001,
launchcf320b97b50fd5e9d6369608d690147cdb1cce9bb15a19d54cfb89bc126f9f1e.
The actual launch recorded81,271,074,816 available RAM bytes,118,629,277,696
artifact-free bytes and3.7% aggregate CPU. One CPU process, one numerical thread;
no new native scene or training. Do not restart or edit its frozen sources.
It must finish through its declared first mission change at1866, validate all
other full decision fields and actual requests, and recheck model/input/source
identities. No outcome from that running replay is claimed here.
