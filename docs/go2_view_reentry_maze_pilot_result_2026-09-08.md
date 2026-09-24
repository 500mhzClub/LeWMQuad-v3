# Translating view recovery native result: navigation failed

Completed native root: `go2_view_reentry_maze_pilot_v1_attempt_001`.
Native result SHA-256:
`0f40eb01e5d5feaf004d0c0e98a9b6d712791dcd676b6013ac965fbb1603ffb8`.
Launch: `cac5f6a64a1a50913e3437133f188089f8e6f7500082ffb9273528300c277e4e`,
1,453 frozen sources. Session 77381 completed in 2,935.967779 seconds.

Completed readout: `go2_view_reentry_maze_readout_v1_attempt_001`.
Result: `7f3105a1864b21f99f24726350d7ffa1636f2ed8ef0f8ce72d849fc29c13fe76`.
Launch: `1b2d3e5c16aae4671d2a690b985baf718ffe24c2aa142a25cdc532a7629cf133`,
1,460 frozen sources. Session 71629 completed. It includes the public-only
auxiliary classification and measured-plane registration diagnosis.

There were 1,517 complete commands, 1,518 paired primary/auxiliary observations,
and 76,600 physics samples. Controller terminal at 1507 was
`NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`, followed by ten
zero commands. There was no physical/acquisition stop, observed arrival, native
arrival window or return. The independent verdict is a failed round trip.

Native XY path length was 7.0802797280761 m. Minimum native outbound goal distance
was 1.271093459247284 m; terminal distance was 1.2768529221576017 m, at initial
XY [3.4722242051375263, -0.09693672063678008]. Nine valid edge crossings include
oscillation at one boundary. The loop-erased outbound route contains five edges:
[-1,0] -> [0,0] -> [0,1] -> [1,1] -> [1,0] -> [2,0]. No invalid edge or
out-of-maze position was reported; maximum native step was 0.000400780381 m.
Maximum observed XY pose error was 0.008476722399662738 m.

The raw physical/public/observer/map/mission/forecast/constraint prefix matches
the predecessor through all 596 observations, including the observation before
the changed command at tick 595. Physical prefix SHA-256:
`593d2984f886977249a90d3a7c44bf2e1003381ce7cc3aece9a4dc18f3496990`.
The actual translating recovery was right_arc [0.16,0,-0.45]. Its raw first
100 ms predicted XY was [0.01265326701104641, 0.0007879026234149933] m;
executed XY was [0.009298246981973414, 0.0017529133246768026] m, error
0.003491046411755475 m. Ordinary nominal selection resumed at tick 597 with a
current-clearance lower bound of 0.4536589108981209 m. This is useful executed
recovery evidence, not a physical-clearance certificate or navigation success.

The final six nominal forecast paths pass; the auxiliary front-left foot contact
check vetoes all actions. The public registration diagnosis in the bound readout
reconstructs the cited samples and their fixed-plane classification failure.
No classification or threshold was changed in this native attempt.

## Separate strict visibility failure

Raw sensor, complete controller/model and command auditing completed, with
model state unchanged. The separate hard-measurement failure list is empty and
all auxiliary frames pass. **Strict physical visibility is false:** primary
frame 909 fails the original sampled check. It compares 4,386 interior rays;
one sampled pixel [row 260, column 428] has native optical depth
1.075383186340332 m versus analytic 2.544169195999424 m, an error of
1.468786009659092 m. The expected object is `novel_wall_5_0_1`.

The native 3x3 neighborhood spans the foreground/background depth edge. This is
an edge-precision hypothesis, not permission to discard the failing ray or
relabel strict visibility. The existing stricter interior/near checks and the
original sampled rule have different results, which must remain explicit.
No pixel was repaired with evaluator geometry. Investigate before a successor
run is used as qualified navigation evidence.

Median observation/control time was 862.2064605 ms, command-inclusive time
893.179496 ms, and receipt-inclusive time 899.1073145 ms. All exceed 100 ms;
physics paused during computation. No real-time or hardware claim is made.

Five native policy attempts have now completed on the same development maze:
four learned controllers and one reactive baseline. All failed navigation;
the latest also fails strict visibility. Zero verified arrivals/round trips and
zero independent-layout executions remain. The next reactive connector native
attempt is separately owned and does not include the new floor-confirmation
candidate.
