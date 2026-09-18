# Direct baseline: collection finished, audit pending

The first-seed full-input direct baseline exhausted its 8,000 navigation-tick
budget without confirming either arrival on independent layout 0. The terminal
mission receipt remains OUTBOUND, with observed goal distance 2.5905 m and
an empty arrivals list. This is a provisional negative collection outcome;
the full sensor and command replay audit is still running in the original
worker, PID 3196770 (creation time 1789286098.79).

Root: `go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_direct_v1_attempt_001`.
The collection has 8,014 paired observations/decisions, 8,013 completed
command ticks, 401,400 physics samples and ten terminal zero ticks. It records
no acquisition stop or physical stop. The top-level result is not yet present.

No run was restarted or shortened. The existing serial continuation remains
active and will proceed after this worker finishes its audit. Do not label
this case fully verified until the terminal audit and source checks finish.
