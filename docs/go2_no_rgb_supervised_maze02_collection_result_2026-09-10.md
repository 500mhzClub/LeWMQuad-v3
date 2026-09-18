# Fifth matched case collected; complete raw audit still pending

The no-RGB supervised maze-2 case completed collection with 3,014 observations,
3,013 executed command intervals and 151,400 physics samples. It ended at
`MISSION_TICK_BUDGET_EXHAUSTED`, followed by the planned terminal zero-command
drain. Collection result SHA-256 is
`6b1189efe0dad05417a80f97a19aa376f722fe0909dded56dabf0b3354d9bbc5`.

An independent readout authenticated the batch launch and completed collection,
physics trace, command tape and timing-stream identities. It reconstructed the
existing native round-trip evaluator and checked every actual command endpoint.
The command tape contains 2,935 zero intervals, 68 left turns and ten right turns;
there are no translating commands. The physical trajectory traversed zero maze
edges, reported zero contact samples and had zero observed arrival claims.
Its maximum XY displacement from the admitted starting pose was 13.790 mm;
terminal displacement was approximately [−3.013, 5.686] mm. The native evaluator
does not identify a round-trip candidate.

[Collection physical readout](go2_no_rgb_supervised_maze02_collection_physical_readout_2026-09-10.json)
has SHA-256
`4f5d9ee042ddef7c6de13c30ceda45af6f77b29f6f3cd15c42fc5b4acdbdf315`.

The original worker remains live as PID 2743870, creation time 1789071424.56.
Its full raw sensor, controller-command and visibility audit has not yet written
an audit report or completed worker/parent receipt. That work must finish under
the original owner. The independently reconstructed physical failure does not
replace the audit, establish strict sensor validity or permit the sixth case to
be launched manually. The audited development count remains 41 at this check.

The earlier fixed-prefix score diagnosis explains why translation was absent:
all actions were reported feasible, but full-plan contact cost outweighed
short-command progress. The saved first-selection contact-horizon probe remains
a diagnostic only; its forward command was never executed in this case. Preserve
this failed physical outcome and the existing queued prospective intervention.
