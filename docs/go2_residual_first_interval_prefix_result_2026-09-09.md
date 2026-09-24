# First-interval feasibility: actual maze2 prefix result

The actual model/RGBD replay completed successfully. It consumed464paired
observations from episode start through463, verified all463prior requested
commands against completed execution, and stopped at its first changed command.
The baseline requested zero motion; the successor selected right_arc
`[0.16,0,-0.45]`. There was no terminal change. No following original observation
was consumed, and the candidate command has not been physically executed.

Root: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_residual_first_interval_prefix_v1_attempt_001`.
Session97118 completed normally, exit0; PID2474093 has finished. Result SHA-256:
`45b9b10e0dc89d4ba477499f817a08df02cc79392c3bf3df9b0363394f7b714b`.
Launch SHA-256:
`64bdc2e18508cc6f1e1386e86ae478156f7a9a192a23398cf0080fa7b8c32283`.
Decision stream SHA-256:
`8fb4bc5e4915df4782432d67fefa88e7e4a87d93dedb30ddea4888af5a3cc2c3`.
Source count1666, final wall573.5599385609385s after launch, including final
verification. Input and source bindings were verified before and after replay.

All461forecast banks available after warmup match exactly. Complete original
selections preserve raw forecasts, scoring and original veto receipts. Observed
state, mission state and causal residual history remain exact. Public input
arrays are unchanged, gradients absent, and assigned model state remains
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.
One fallback was attempted; the first selected-action change and first requested
command change are both463. The run stops there rather than treating the old
episode's later recovery/failure as an outcome of the new right arc.

At the intervention, the current nominal observed clearance is
0.45751049799042826m against radius0.45m. The existing online residual correction
is[-0.0038673035867011927,0.0049220819208829775]m. Right_arc's raw first-stepXY
is[0.005903249606490135,0.007221844047307968]m; its corrected first-stepXY is
[0.009770553193191329,0.0022997621264249906]m. Raw first-segment clearance is
0.4497106959665005m. The corrected eight-segment path has minimum nominal
clearance0.45062258240705394m, and both original and corrected first-step
surface checks pass. Right_arc is the only eligible corrected action, with
original corrected100ms utility−0.01278053472898414m.

The0.623mm minimum nominal excess over the radius is not a model-error bound,
physical-clearance certificate or proof of successful movement. Later forecast
points, yaw/contact predictions and original residual-learning targets remain
unchanged. This result proves a causal prospective command difference on an
inspected development episode, not that maze2's eventual503stop is fixed.

Next: prepare a separate fresh native maze2 successor, preserving the exact
source/model/protocol and comparing the full original physical/public prefix
through observation463, plus complete candidate decisions against this replay.
That prefix consists of464paired observation endpoints and23900physics samples
(last endpoint index23899). Execute the candidate command only in that fresh
run; audit its actual observations, commands, visibility and native trajectory.
Do not change the active reactive cohort or the queued planning-memory comparison.
Keep this prefix and its sources fixed.

Hardware after completion:80.169GBavailableRAM,90.849GBartifactfree,
21.360GBworkspacefree,CPU3.3%,bothGPUs0%,all32CPUaffinity. The only substantive
remaining Python job was independent-reactive admissionPID2475541. There is
still no navigation, timing, hardware or deployment qualification.
