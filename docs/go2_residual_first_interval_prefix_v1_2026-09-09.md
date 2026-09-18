# First-interval feasibility: actual observation prefix V1

Execute one ordered CPU replay from episode start on completed development
maze2, using the exact assigned full-JEPA rollout model and original paired
RGB/depth, body and control histories. No training, native scene or command
execution. This is a prospective command comparison on an already inspected
development failure, not a held-out evaluation or physical outcome claim.

Input cohort result SHA-256:
`a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720`.
Case `full_jepa_novel_maze_02`, model `seed_2026091001_full_jepa`, assigned
state `4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.
Use the original correction admission. Authenticate completed raw audits and
exact source/runtime/input bindings before loading; recheck after replay.

The only controller change is ResidualFirstIntervalController, defined in
`docs/go2_residual_first_interval_feasibility_preparation_2026-09-09.md`.
Compare complete original selections, including every forecast, score and
original veto receipt. Compare every other decision field exactly except the
explicit controller identifier/enable flag and declared action/wait accounting
after a fallback changes the selected action. Observed mapping, registration,
mission and preceding residuals must remain exact. At a changed terminal only
the current pending-forecast marker may additionally differ.

Stop at the first changed requested command, changed terminal, original
terminal, or504 observations (indices0–503), whichever comes first. Do not
consume the following observation or decision. A fallback that changes an
infeasible zero request to an eligible hold may continue only while actual
requests remain identical; report its first selected-action change separately.
All prior requested commands must equal the originally dispatched completed
commands. Inputs may not be mutated; weights and absent gradients are checked
afterward. Failure retains the exclusive attempt without automatic retry.

Output: `go2_residual_first_interval_prefix_v1_attempt_001` under the existing
owned navigation-development artifact root. Bind source inventory, protocol,
component/test preparation and input identities in launch.json. Stream complete
decisions with deterministic gzip. Reserve256MiB output, require8GiB available
RAM plus the existing40GiB artifact reserve, and stop at128MiB compressed output
to retain receipt/failure headroom. Resource checks are capacity admissions,
not OS-enforced limits. One OpenCV/BLAS thread and one sequential replay worker;
this CPU replay may run beside the separately owned reactive native scene after
hardware refresh. Never create a second native scene for this replay.

Successful replay proves only the checked causal prefix and prospective command.
Fresh native execution, complete raw audit and physical evaluation remain
necessary to assess navigation effects. Preserve every baseline failure and
the previously prepared reactive and planning-memory comparison definitions.
