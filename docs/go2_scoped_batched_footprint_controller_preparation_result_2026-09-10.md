# Combined footprint-query optimization prepared

The new controller combines two existing implementations: the batched patch
controller's fresh primary and auxiliary patch stores and the scoped selector's
exact query reuse. The measured-memory type remains unchanged, so reuse enters
its supported branch. No observation update, mission, forecast, feasibility
rule, model correction or retained-history limit is changed.

All 79 focused tests passed in 8.87 seconds (session 61992, exit zero). The new
composition tests compare synthetic observed-memory receipts using the actual
Go2 articulated collision geometry, verify both camera patch stores are queried,
verify cache hits and cleanup, and preserve independent public receipt ownership.
A separate synthetic 1,428-frame history preserves the earliest complete witness
at frame 1418 and retains uncovered results. Retained-state comparison normalizes
only the two declared patch-store type tags. These are component and composition
tests, not trained-model inference or recorded native-trajectory replay.

Source preparation verified 2,149 bindings and authenticated the existing batched
405-frame replay result and its outputs. That earlier isolated batching experiment
reduced total controller time by 4.95%, with an early-navigation regression.
Its result does not measure the combination's benefit.

[Preparation record](go2_scoped_batched_footprint_controller_preparation_2026-09-10.json)
has SHA-256
`b8a9c8b332838e5422d67b9ba15daf0414c9999a51ee8b3b687b108acdc88e13`.

The scoped-only 1,428-frame paired replay remains live as PID 2754886, creation
time 1789077365.71. Finish and authenticate that attempt, then inspect its timing
and equivalence evidence before preparing a separately named full incremental
comparison for this combination. That combined replay is not yet implemented.
Do not restart the current replay because its initial input admission is quiet.

The ongoing simulator batch remains at four completed cases; the fifth case
had completed timing row 2558 during this turn. No simulator policy was changed,
no scene was added, and no new navigation success or real-time result follows
from this preparation.
