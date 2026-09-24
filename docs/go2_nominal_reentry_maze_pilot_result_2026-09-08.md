# Nominal-reentry maze pilot: completed, navigation failed

Native root `go2_nominal_reentry_maze_pilot_v1_attempt_001` completed with
result SHA-256 `b48f3c79d19889c67438073a3a7305d0eafcbc5f8b7286ea014e2eb1fd711739`.
Launch SHA-256 `676c13ed551a986ee5669503bcac961de63d564dc932be9dbaabe5993ee37c59`
binds 1,419 source paths. Session 58272 completed; its processes terminated.
This reused development maze 0 with the unchanged JEPA seed 2026091001 model.
It adds zero independent layouts and zero verified arrivals or round trips.

Collection completed 442 commands, 443 paired observations and 22,850 physics
samples. The controller stopped at decision 432 for no phase-admissible candidate
satisfying surface and nominal constraints, followed by ten zero drain commands.
No physical or acquisition stop occurred. Raw sensor reconstruction, model replay,
actual command audit, unchanged model state and physical visibility all passed;
no hard measurement failure was recorded. The physical and public prefix matched
all 408 observations and 21,100 physics samples before changed command 407.

Readout root `go2_nominal_reentry_maze_readout_v1_attempt_001` completed with
result SHA-256 `0ff22595c9a49fb855212952a11f02915a6269c22db3011180c03dbfda0a20f7`.
Its launch SHA-256 is `e7b5f57741fc50c7b96dd067a03169f62151e72e9414f0a985eb7ddfe1bc9c96`,
with 1,423 bound sources. Session 90071 completed. Two focused readout tests
passed in 1.70 s, covering body-frame execution, incomplete-interval censoring,
command mismatch rejection and ordinary nominal-gate transition evidence.

The only nominal-reentry actions were left turns at 407, 408 and 409. Predicted
first-step clearance gains were respectively 0.414, 1.810 and 3.028 mm; actual
native body-XY forecast errors were 6.583, 5.590 and 7.076 mm. At decision 410
an ordinary right-turn selection passed the unchanged 0.45 m nominal gate;
its first segment's minimum observed clearance was 0.451830280 m. This is
observed-map reentry evidence, not physical clearance certification. Reentry
therefore occurred, but did not solve continued execution: at 422 the observed
current clearance fell to 0.447522694 m; no further recovery action was eligible.

Native traversal remained `[-1,0] -> [0,0] -> [0,-1]`, the same two open edges
as the first pilot. Path length was 2.456451478 m; minimum outbound-goal distance
2.768958301 m and terminal native distance 3.019180489 m. No outbound or return
arrival window exists. Maximum observed-pose XY error was 5.260 mm. Median
observation/control took 749.382 ms, command-inclusive execution 780.992 ms and
receipt-inclusive iteration 786.404 ms. All cycles exceeded 100 ms while physics
paused. Neither real-time nor hardware qualification is established.

Readout admission recorded 82.902 GB available RAM, 55.288 GB free artifact space,
CPU 0.4% busy and both GPUs idle. One ordered CPU readout ran; no scene or model
training was launched. The full goal remains active. Matched reactive,
non-predictive and learned ablations, other independent layouts, verified
backtracking and realistic timing/sensing remain incomplete.
