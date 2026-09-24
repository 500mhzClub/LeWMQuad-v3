# Common-floor registered native maze pilot: completed failure

The eighth native maze-0 attempt and its readout are complete. Native result
`f7a6ef564d6e2bfac8b259e2ddc1988e5b357d8be7f180512f8b00393bbdfe41`, root
`go2_joint_floor_registered_maze_pilot_v1_attempt_001`. Launch
`7478a094256d99aa3c25958806707730efb07eb3277dd83e411437b7d9ee98a5` binds1509
sources. Session90647 closed successfully. Wall2628.965827s; worker2608.244913s.

Collection1067commands,1068paired observations,54100physics samples. Controller
terminal1057 for no candidate satisfying surface and nominal constraints, then
10zero drain commands. No physical/acquisition stop or observed/native arrival
or return. All raw sensor/model/command audits passed; weights remained unchanged.
Strict physical visibility failed; the hard measurement failure list is empty.
The completed native audit preserves that failed qualification outcome.

Actual physics and public observations matched the bound predecessor for960
frames through the intervention observation959. Every candidate decision in
that prefix matched the prospective common-floor replay exactly. Physics prefix
SHA-256 `d9311f66e3d861870f9bf021979aa75d931b089e967d5af031cd704a17eab5d3`.

Readout session1811 complete, root
`go2_joint_floor_registered_maze_readout_v1_attempt_001`; result SHA-256
`5665f51fbb5315d2467494b295ef3c8ba4f4eecacb1629c5b70bdd6b78ae0390`.
Readout launch `2964215c3be11c74268a251eb818fc9c1c60be571d761280d3d34b37b70d505b`,1513sources.

- Native XY path4.669673281m; minimum and terminal goal distance2.886041376m.
- Seven valid open-edge crossings, three after loop erasure; all positions in
  the maze and no invalid crossing. Return never began.
- 1058 admitted poses; terminal drain1058–1067 has no new pose, explicitly
  omitted from pose accuracy rather than filled in.
- Raw/registered mean XYZ error4.27175/3.57577mm, maximum8.30124/7.24357mm.
- Raw/registered mean XY error3.58814/3.57408mm, maximum7.34253/7.24060mm.
- Raw/registered mean rotation error0.003248385/0.002136547rad,
  maximum0.005101844/0.003850614rad.
- Median receipt-inclusive loop1163.238353ms per100ms paused-physics step.

Registration improved these pose-error summaries on the actual executed
trajectory but did not achieve navigation. This is not a paired navigation
advantage claim or a calibrated pose bound. Full terminal diagnosis and later
floor-view evidence are recorded separately in
`go2_joint_floor_registered_terminal_provisional_diagnosis_2026-09-09.md` and
`go2_retained_floor_resolution_diagnosis_result_2026-09-09.md`; their collection
identities are now covered by this completed native result. Those historical
diagnostics retain their provisional labels to preserve chronology.

Eight completed native maze-0 attempts are failures. Zero verified arrivals
or round trips; independent mazes1–3, matched baselines, real-time execution
and hardware validation remain outstanding. The full goal remains active.
